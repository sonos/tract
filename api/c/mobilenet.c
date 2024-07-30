#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>  // for gettimeofday()
#include <tract.h>

#define check(call) {                                                           \
    TRACT_RESULT result = call;                                                 \
    if(result == TRACT_RESULT_KO) {                                             \
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());     \
        exit(1) ;                                                               \
    }                                                                           \
}

typedef struct prediction {
    double pred;
    int category;
    TractValue *output;
}prediction;

prediction **
init_predictions(int arg)
{
    prediction **inf = malloc(arg * sizeof(prediction));
    if (!inf) {
        fprintf(stderr, "Error allocating memory for predictions\n");
        return NULL;
    }

    for (int i = 0; i < arg; i++) {
        inf[i] = malloc(sizeof(prediction));
        if (!inf[i]) {
            fprintf(stderr, "Error allocating memory for prediction\n");
            return NULL;
        }
        inf[i]->pred = 0.0;
        inf[i]->category = 0;
        inf[i]->output = NULL;
    }
    return inf;
}

void
free_prediction(prediction *inf)
{
    if (!inf) {
        return;
    }
    if (inf->output) {
        check(tract_value_destroy(&inf->output));
        assert(!inf->output);
    }
    free(inf);
}

void
free_predictions(prediction **inf, int length)
{
    for (int i = 0; i < length; i++) {
        free_prediction(inf[i]);
    }
    free(inf);
}

prediction *
inference(char *model_name, TractValue *input, TractValue *input2, prediction *inf)
{
    struct timeval t1, t2;
    double elapsedTime;

    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check(tract_onnx_create(&onnx));
    assert(onnx);

    // Load the model
    TractModel *model = NULL;
    TractInferenceModel *inference_model = NULL;
    if (tract_onnx_model_for_path(onnx, model_name, &inference_model,NULL) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error loading model %s\n", model_name);
        free_prediction(inf);
        check(tract_onnx_destroy(&onnx));
        assert(!onnx);
        return NULL;
    }
    assert(inference_model);
    assert(onnx);

    check(tract_onnx_destroy(&onnx));
    assert(!onnx);

    // Convert inference model to a typed model and optimize it
    check(tract_inference_model_into_optimized(&inference_model,&model));
    assert(model);

    // Make the model runnable
    TractRunnable *runnable = NULL;
    check(tract_model_into_runnable(&model, &runnable));
    assert(runnable);
    assert(!model);

    TractValue* output = NULL;

    gettimeofday(&t1, NULL);

    // simple stateless run...
    TractValue *inputs[] = { input, input2 };
    check(tract_runnable_run(runnable, inputs, &output));

    const float *data = NULL;
    check(tract_value_as_bytes(output, NULL, NULL, NULL, (const void**) &data));

    check(tract_runnable_release(&runnable));
    assert(!runnable);

    float max = data[0];
    int argmax = 0;
    for(int i = 0; i < 1000; i++) {
        float val = data[i];
        if(val > max) {
            max = val;
            argmax = i;
        }
    }
    assert(data[argmax] == max);
    fprintf(stderr, "\nModel: %s\nMax is %f for category %d\n", model_name, max, argmax);

    
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("Model run in %f ms.\n", elapsedTime);

    inf->output = output;
    inf->pred = max;
    inf->category = argmax;

    return inf;
}

size_t *
decode_pb(FILE *fd)
{
    uint8_t byte;
    uint8_t wire_type;
    uint64_t varint_value;
    static size_t shape[4] = {0, 0, 0, 0};

    int k=0;
    for (int k = 0; k < 4; k++) {
        fread(&byte, sizeof(uint8_t), 1, fd);
        wire_type = byte & 0x07;

        if (wire_type == 0) {
            varint_value = 0;
            int shift = 0;
            do {
                fread(&byte, sizeof(uint8_t), 1, fd);
                varint_value |= (uint64_t)(byte & 0x7F) << (7 * shift);
                shift++;
            } while (byte & 0x80);
            shape[k] = varint_value;
        } else {
            break;
        }
    }

    fseek(fd, 0, SEEK_SET);

    return shape;
}

int
main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model1.onnx> <model2.onnx> ... <modelN.onnx> <input.pb>\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc-1; i++) {
        FILE *fd = fopen(argv[i], "rb");
        if (!fd) {
            fprintf(stderr, "Error opening model file %s\n", argv[i]);
            return 1;
        }
        fclose(fd);
    }

    FILE *fd = fopen(argv[argc - 1], "rb");
    if (!fd) {
        fprintf(stderr, "Error opening model_input file");
        return 1;
    }
    
    size_t *shape = decode_pb(fd);
    int calculated_shape = shape[0] * shape[1] * shape[2] * shape[3];
    fprintf(stderr, "Input shape: %zu %zu %zu %zu\n", shape[0], shape[1], shape[2], shape[3]);

    float *image = (float *) malloc( calculated_shape * sizeof(float));
    int image_floats = fread(image, sizeof(float), calculated_shape, fd);
    fprintf(stderr, "Read %d floats\n", image_floats);
    assert(image_floats == calculated_shape);
    fclose(fd);

    prediction** preds = init_predictions(argc-1);
    if (!preds) {
        return 1;
    }

    check(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, 4, shape, image, &preds[0]->output));
    free(image);

    //Hint for splitting the models into a node that is part of cut from parent node (circle)
    //The inference of the last model is gonna take the output of the 2 previous models, like input2, input3
    for (int i = 1; i < argc-1; i++) {
        preds[i] = inference(argv[i], preds[i-1]->output, NULL, preds[i]);
    }

    free_predictions(preds, argc-1);
    fprintf(stderr, "All done\n");

    return 0;
}
