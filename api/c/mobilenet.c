#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "tract.h"

#define check(call) {                                                             \
    TRACT_RESULT result = call;                                                 \
    if(result == TRACT_RESULT_KO) {                                             \
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());     \
        exit(1) ;                                                               \
    }                                                                           \
}                                                                               \

int main() {
    // Initialize nnef parser
    TractNnef *nnef = NULL;
    check(tract_nnef_create(&nnef));
    assert(nnef);

    // Load the model
    TractModel *model = NULL;
    check(tract_nnef_model_for_path(nnef, "mobilenet_v2_1.0.onnx.nnef.tgz", &model));
    assert(model);
    assert(nnef);

    // once the model is build, the framework is not necessary anymore
    check(tract_nnef_destroy(&nnef));
    assert(!nnef);

    // Optimize the model
    check(tract_model_optimize(model));

    // Make the model runnable
    TractRunnable *runnable = NULL;
    check(tract_model_into_runnable(&model, &runnable));
    assert(runnable);
    assert(!model);

    float *image = malloc(3*224*224*sizeof(float));
    FILE *fd = fopen("grace_hopper_3_224_224.f32.raw", "rb");
    assert(fread(image, sizeof(float), 3*224*224, fd) == 3*224*224);
    fclose(fd);

    TractValue* input = NULL;
    size_t shape[] = {1, 3, 224, 224 };
    check(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, 4, shape, image, &input));
    free(image);

    TractValue* output = NULL;

    // simple stateless run...
    check(tract_runnable_run(runnable, &input, &output));

    const float *data = NULL;
    check(tract_value_as_bytes(output, NULL, NULL, NULL, (const void**) &data));
    float max = data[0];
    int argmax = 0;
    for(int i = 0; i < 1000 ; i++) {
        float val = data[i];
        if(val > max) {
            max = val;
            argmax = i;
        }
    }
    printf("Max is %f for category %d\n", max, argmax);
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
    check(tract_value_destroy(&output));

    // done with out state and input
    check(tract_state_destroy(&state));
    check(tract_value_destroy(&input));
}

