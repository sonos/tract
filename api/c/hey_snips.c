#include "tract.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define check(call)                                                            \
    {                                                                          \
        TRACT_RESULT result = call;                                            \
        if (result == TRACT_RESULT_KO) {                                       \
            fprintf(stderr, "Error calling tract: %s",                         \
                    tract_get_last_error());                                   \
            exit(1);                                                           \
        }                                                                      \
    }

float *read_speech(size_t *len) {
    FILE *fd = fopen("hey-snips-tts.wav", "rb");
    assert(fd);
    fseek(fd, 0, SEEK_END);
    size_t size = ftell(fd);
    fseek(fd, 0, SEEK_SET);
    uint8_t *wav = malloc(size);
    assert(fread(wav, 1, size, fd) == size);
    fclose(fd);
    assert(memcmp(wav + 36, "data", 4) == 0);
    *len = (size - 44) / 2;
    float *samples = malloc(*len * sizeof(float));
    for (size_t i = 0; i < *len; i++) {
        int16_t sample = (int16_t)(wav[44 + 2 * i] | wav[45 + 2 * i] << 8);
        samples[i] = sample / 32768.0f;
    }
    free(wav);
    return samples;
}

size_t run_pulses(TractState *state, const float *speech, size_t pulse,
                  size_t count, float *scores) {
    size_t frames = 0;
    for (size_t p = 0; p < count; p++) {
        TractTensor *input = NULL;
        size_t shape[] = {pulse};
        check(tract_tensor_from_bytes(TRACT_DATUM_TYPE_F32, 1, shape,
                                      (void *)(speech + p * pulse), &input));
        TractTensor *output = NULL;
        check(tract_state_run(state, &input, &output));
        check(tract_tensor_destroy(&input));
        uintptr_t rank = 0;
        const uintptr_t *output_shape = NULL;
        const float *logits = NULL;
        check(tract_tensor_as_bytes(output, NULL, &rank, &output_shape,
                                    (const void **)&logits));
        size_t values = 1;
        for (size_t axis = 0; axis < rank; axis++) {
            values *= output_shape[axis];
        }
        for (size_t i = 0; i < values / 2; i++) {
            scores[frames++] = logits[2 * i + 1] - logits[2 * i];
        }
        check(tract_tensor_destroy(&output));
    }
    return frames;
}

int main() {
    TractNnef *nnef = NULL;
    check(tract_nnef_create(&nnef));
    check(tract_nnef_enable_pulse(nnef));
    TractModel *model = NULL;
    check(tract_nnef_load(nnef, "hey_snips.nnef.tar", &model));
    check(tract_nnef_destroy(&nnef));

    TractFact *fact = NULL;
    check(tract_model_input_fact(model, 0, &fact));
    TractDim *dim = NULL;
    check(tract_fact_dim(fact, 0, &dim));
    int64_t pulse = 0;
    check(tract_dim_to_int64(dim, &pulse));
    check(tract_dim_destroy(&dim));
    check(tract_fact_destroy(&fact));

    TractTensor *property = NULL;
    check(tract_model_property(model, (const int8_t *)"pulse.delay", &property));
    const int64_t *delay = NULL;
    check(tract_tensor_as_bytes(property, NULL, NULL, NULL,
                                (const void **)&delay));
    int64_t skip = delay[0];
    check(tract_tensor_destroy(&property));

    TractRunnable *runnable = NULL;
    check(tract_model_into_runnable(&model, &runnable));
    TractState *state = NULL;
    check(tract_runnable_spawn_state(runnable, &state));
    check(tract_runnable_release(&runnable));

    uintptr_t count = 0;
    check(tract_state_input_count(state, &count));
    assert(count == 1);
    check(tract_state_output_count(state, &count));
    assert(count == 1);

    size_t len = 0;
    float *speech = read_speech(&len);
    size_t pulses = len / pulse;
    size_t first = pulses / 2;
    float *scores = malloc(len * sizeof(float));
    float *clone_scores = malloc(len * sizeof(float));
    size_t frames = run_pulses(state, speech, pulse, first, scores);

    TractState *clone = NULL;
    check(tract_state_clone(state, &clone));
    const float *rest = speech + first * pulse;
    size_t rest_frames =
        run_pulses(state, rest, pulse, pulses - first, scores + frames);
    size_t clone_frames =
        run_pulses(clone, rest, pulse, pulses - first, clone_scores);
    assert(clone_frames == rest_frames);
    assert(memcmp(scores + frames, clone_scores,
                  rest_frames * sizeof(float)) == 0);
    frames += rest_frames;
    check(tract_state_destroy(&clone));
    check(tract_state_destroy(&state));

    int wake_ups = 0;
    for (size_t i = skip + 1; i < frames; i++) {
        if (scores[i - 1] <= 2.0 && scores[i] > 2.0) {
            wake_ups++;
        }
    }
    free(speech);
    free(scores);
    free(clone_scores);

    printf("%d wake ups\n", wake_ups);
    assert(wake_ups == 4);
}
