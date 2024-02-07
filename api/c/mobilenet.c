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
}                                                                               \

TractValue *
inference(char *model_name, TractValue *input, TractValue *input2)
{
    struct timeval t1, t2;
    double elapsedTime;

    // Initialize nnef parser
    TractOnnx *onnx = NULL;
    check(tract_onnx_create(&onnx));
    assert(onnx);

    // Load the model
    TractModel *model = NULL;
    TractInferenceModel *inference_model = NULL;
    check(tract_onnx_model_for_path(onnx, model_name, &inference_model));
    assert(inference_model);
    assert(onnx);

    // once the model is build, the framework is not necessary anymore
    check(tract_onnx_destroy(&onnx));
    assert(!onnx);

    // Convert inference model to a typed model
    check(tract_inference_model_into_typed(&inference_model, &model));
    assert(model);

    //Optimize the model
    check(tract_model_optimize(model));
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

    float max = data[0];
    int argmax = 0;
    for(int i = 0; i < 1000; i++) {
        float val = data[i];
        if(val > max) {
            max = val;
            argmax = i;
        }
    }
    printf("\nModel: %s\nMax is %f for category %d\n", model_name, max, argmax);
    check(tract_value_destroy(&output));

    // or spawn a state to run the model
    TractState *state = NULL;
    check(tract_runnable_spawn_state(runnable, &state));
    assert(state);

    // runnable is refcounted by the spawned states, so we can release it now.
    check(tract_runnable_release(&runnable));
    assert(!runnable);

    check(tract_state_run(state, &input, &output));

    check(tract_value_as_bytes(output, NULL, NULL, NULL, (const void**) &data));
    assert(data[argmax] == max);

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("Model run in %f ms.\n", elapsedTime);

    return output;
}


int main(int argc, char **argv) {
    size_t shape[4] = {1, 3, 224, 224};

    int calculated_shape = shape[1] * shape[2] * shape[3];
    float *image = malloc(calculated_shape*sizeof(float));
    FILE *fd = fopen(argv[4], "rb");
    assert(fread(image, sizeof(float), calculated_shape, fd) == calculated_shape);
    fclose(fd);

    TractValue* input = NULL;
    check(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, 4, shape, image, &input));
    free(image);

    TractValue* input1 = inference(argv[1], input, NULL);

    TractValue* input2 = inference(argv[2], input, NULL);
    check(tract_value_destroy(&input));

    TractValue* input3 = inference(argv[3], input2, NULL);

    //third model
    //TractValue* input4 = inference("../../../cONNXr/test/tiny_yolov2/model_2.onnx", input3, NULL);
    //TractValue* input4 = inference("../../../cONNXr/test/mobilenetv2-1.0/mobilenetv2-1.0_test_22.onnx", input2, input3);
    //check(tract_value_destroy(&input4));
    

    check(tract_value_destroy(&input1));
    check(tract_value_destroy(&input3));
    check(tract_value_destroy(&input2));

    return 0;
}
