#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "dlpack.h"
#include "tract.h"

void dump_tensor(DLTensor *tensor) {
    int len = 1;
    for(int i = 0; i < tensor->ndim; i++) {
        len *= tensor->shape[i];
        fprintf(stdout, "%d ", tensor->shape[i]);
    }
    fprintf(stdout, "= ");
    for(int i = 0; i < len; i++) {
        fprintf(stdout, "%f ", ((float*)(tensor->data))[i]);
    }
    fprintf(stdout, "\n");
}

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

    // input def and output decl
    float data[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    int64_t shape[3] = { 2, 1, 3 };
    DLDataType f32 = { .code = kDLFloat, .bits = 32, .lanes = 1 };
    DLTensor input = {
        .data = data, .device = kDLCPU, .ndim = 3, .dtype = f32, .shape = shape, .strides = NULL, .byte_offset = 0
    };
    DLTensor output;
    memset(&output, 0, sizeof(DLTensor));

    // simple stateless run...
    check(tract_runnable_run(runnable, 1, &input, 1, &output));
    dump_tensor(&output);

    // or spawn a state to run the model
    TractState *state = NULL;
    check(tract_runnable_spawn_state(runnable, &state));
    assert(state);

    // runnable is refcounted by the spawned states, so we can release ours.
    check(tract_runnable_release(&runnable));
    assert(!runnable);

    memset(&output, 0, sizeof(DLTensor));
    check(tract_state_set_input(state, 0, &input));
    check(tract_state_exec(state));
    check(tract_state_output(state, 0, &output));

    dump_tensor(&output);

    // done with out state
    check(tract_state_destroy(&state));
}

