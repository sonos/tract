#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>  // for gettimeofday()
#include "tract.h"

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
    for (int i = 0; i < arg; i++) {
        inf[i] = malloc(sizeof(prediction));
        inf[i]->pred = 0.0;
        inf[i]->category = 0;
        inf[i]->output = NULL;
    }
    return inf;
}

void
free_predictions(prediction **inf, int arg)
{
    for (int i = 0; i < arg; i++) {
        check(tract_value_destroy(&inf[i]->output));
        assert(!inf[i]->output);
        free(inf[i]);
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
    check(tract_onnx_model_for_path(onnx, model_name, &inference_model));
    assert(inference_model);
    assert(onnx);

    check(tract_onnx_destroy(&onnx));
    assert(!onnx);

    // Convert inference model to a typed model and optimize it
    // check(tract_inference_model_into_typed(&inference_model, &model));
    // assert(model);
    // check(tract_model_optimize(model));
    // assert(model);

    // or
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
    printf("\nModel: %s\nMax is %f for category %d\n", model_name, max, argmax);

    // or spawn a state to run the model
    // TractState *state = NULL;
    // check(tract_runnable_spawn_state(runnable, &state));
    // assert(state);

    // check(tract_state_run(state, &input, &output));

    // check(tract_value_as_bytes(output, NULL, NULL, NULL, (const void**) &data));
    // check(tract_state_destroy(&state));
    // assert(!state);

    // printf("\nModel: %s\nMax is %f for category %d\n", model_name, max, argmax);

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("Model run in %f ms.\n", elapsedTime);

    inf->output = output;
    inf->pred = max;
    inf->category = argmax;

    return inf;
}


int
main(int argc, char **argv)
{
    char command[256];
    snprintf(command, sizeof (command), "python3 get_size_io.py %s", argv[1]);

    FILE *cmd = popen(command, "r");
    char result[100];
    while (fgets(result, sizeof(result), cmd) !=NULL) {}
    pclose(cmd);

    size_t shape[4];
    char *token;

    // Skip "name of first node: " prefix
    token = strtok(result, ":");
    token = strtok(NULL, ":");
    token = strtok(token, " ");
    size_t i = 0;
    while (token) {
        char *endptr;
        long number = strtol(token, &endptr, 10);
        shape[i++] = number;
        if (number == 0) {
            break;
        }

        token = strtok(NULL, " ");
    }
    fprintf(stderr, "\nShape: %zu, %zu, %zu, %zu\n", shape[0], shape[1], shape[2], shape[3]);

    int calculated_shape = shape[0] * shape[1] * shape[2] * shape[3];
    float *image = malloc(calculated_shape*sizeof(float));
    FILE *fd = fopen(argv[argc - 1], "rb");
    assert(fread(image, sizeof(float), calculated_shape, fd) == calculated_shape);
    fclose(fd);

    prediction** preds = init_predictions(argc-1);

    check(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, 4, shape, image, &preds[0]->output));
    free(image);

    //If model is cut inside a circle, the inference of the last model is gonna take the output of the 2 previous models, like input2, input3
    for (int i = 1; i < argc-1; i++) {
        preds[i] = inference(argv[i], preds[i-1]->output, NULL, preds[i]);
    }

    free_predictions(preds, argc-1);

    return 0;
}
