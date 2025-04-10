#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
    typedef enum DatumType {
      TRACT_DATUM_TYPE_BOOL = 1,
      TRACT_DATUM_TYPE_U8 = 17,
      TRACT_DATUM_TYPE_U16 = 18,
      TRACT_DATUM_TYPE_U32 = 20,
      TRACT_DATUM_TYPE_U64 = 24,
      TRACT_DATUM_TYPE_I8 = 33,
      TRACT_DATUM_TYPE_I16 = 34,
      TRACT_DATUM_TYPE_I32 = 36,
      TRACT_DATUM_TYPE_I64 = 40,
      TRACT_DATUM_TYPE_F16 = 50,
      TRACT_DATUM_TYPE_F32 = 52,
      TRACT_DATUM_TYPE_F64 = 56,
      TRACT_DATUM_TYPE_COMPLEX_I16 = 66,
      TRACT_DATUM_TYPE_COMPLEX_I32 = 68,
      TRACT_DATUM_TYPE_COMPLEX_I64 = 72,
      TRACT_DATUM_TYPE_COMPLEX_F16 = 82,
      TRACT_DATUM_TYPE_COMPLEX_F32 = 84,
      TRACT_DATUM_TYPE_COMPLEX_F64 = 88,
    } DatumType;


/**
 * Used as a return type of functions that can encounter errors.
 * If the function encountered an error, you can retrieve it using the `tract_get_last_error`
 * function
 */
typedef enum TRACT_RESULT {
  /**
   * The function returned successfully
   */
  TRACT_RESULT_OK = 0,
  /**
   * The function returned an error
   */
  TRACT_RESULT_KO = 1,
} TRACT_RESULT;

typedef struct TractFact TractFact;

typedef struct TractInferenceFact TractInferenceFact;

typedef struct TractInferenceModel TractInferenceModel;

typedef struct TractModel TractModel;

typedef struct TractNnef TractNnef;

typedef struct TractOnnx TractOnnx;

typedef struct TractRunnable TractRunnable;

typedef struct TractState TractState;

typedef struct TractValue TractValue;

/**
 * Retrieve the last error that happened in this thread. A function encountered an error if
 * its return type is of type `TRACT_RESULT` and it returned `TRACT_RESULT_KO`.
 *
 * # Return value
 *  It returns a pointer to a null-terminated UTF-8 string that will contain the error description.
 *  Rust side keeps ownership of the buffer. It will be valid as long as no other tract calls is
 *  performed by the thread.
 *  If no error occured, null is returned.
 */
const char *tract_get_last_error(void);

/**
 * Returns a pointer to a static buffer containing a null-terminated version string.
 *
 * The returned pointer must not be freed.
 */
const char *tract_version(void);

/**
 * Frees a string allocated by libtract.
 */
void tract_free_cstring(char *ptr);

/**
 * Creates an instance of an NNEF framework and parser that can be used to load and dump NNEF models.
 *
 * The returned object should be destroyed with `tract_nnef_destroy` once the model
 * has been loaded.
 */
enum TRACT_RESULT tract_nnef_create(struct TractNnef **nnef);

enum TRACT_RESULT tract_nnef_transform_model(const struct TractNnef *nnef,
                                             struct TractModel *model,
                                             const char *transform_spec);

enum TRACT_RESULT tract_nnef_enable_tract_core(struct TractNnef *nnef);

enum TRACT_RESULT tract_nnef_enable_tract_extra(struct TractNnef *nnef);

enum TRACT_RESULT tract_nnef_enable_tract_transformers(struct TractNnef *nnef);

enum TRACT_RESULT tract_nnef_enable_onnx(struct TractNnef *nnef);

enum TRACT_RESULT tract_nnef_enable_pulse(struct TractNnef *nnef);

enum TRACT_RESULT tract_nnef_enable_extended_identifier_syntax(struct TractNnef *nnef);

/**
 * Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
 */
enum TRACT_RESULT tract_nnef_destroy(struct TractNnef **nnef);

/**
 * Parse and load an NNEF model as a tract TypedModel.
 *
 * `path` is a null-terminated utf-8 string pointer. It can be an archive (tar or tar.gz file) or a
 * directory.
 */
enum TRACT_RESULT tract_nnef_model_for_path(const struct TractNnef *nnef,
                                            const char *path,
                                            struct TractModel **model);

/**
 * Dump a TypedModel as a NNEF tar file.
 *
 * `path` is a null-terminated utf-8 string pointer to the `.tar` file to be created.
 *
 * This function creates a plain, non-compressed, archive.
 */
enum TRACT_RESULT tract_nnef_write_model_to_tar(const struct TractNnef *nnef,
                                                const char *path,
                                                const struct TractModel *model);

/**
 * Dump a TypedModel as a NNEF .tar.gz file.
 *
 * `path` is a null-terminated utf-8 string pointer to the `.tar.gz` file to be created.
 */
enum TRACT_RESULT tract_nnef_write_model_to_tar_gz(const struct TractNnef *nnef,
                                                   const char *path,
                                                   const struct TractModel *model);

/**
 * Dump a TypedModel as a NNEF directory.
 *
 * `path` is a null-terminated utf-8 string pointer to the directory to be created.
 *
 * This function creates a plain, non-compressed, archive.
 */
enum TRACT_RESULT tract_nnef_write_model_to_dir(const struct TractNnef *nnef,
                                                const char *path,
                                                const struct TractModel *model);

/**
 * Creates an instance of an ONNX framework and parser that can be used to load models.
 *
 * The returned object should be destroyed with `tract_nnef_destroy` once the model
 * has been loaded.
 */
enum TRACT_RESULT tract_onnx_create(struct TractOnnx **onnx);

/**
 * Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
 */
enum TRACT_RESULT tract_onnx_destroy(struct TractOnnx **onnx);

/**
 * Parse and load an ONNX model as a tract InferenceModel.
 *
 * `path` is a null-terminated utf-8 string pointer. It must point to a `.onnx` model file.
 */
enum TRACT_RESULT tract_onnx_model_for_path(const struct TractOnnx *onnx,
                                            const char *path,
                                            struct TractInferenceModel **model);

/**
 * Query an InferenceModel input counts.
 */
enum TRACT_RESULT tract_inference_model_input_count(const struct TractInferenceModel *model,
                                                    uintptr_t *inputs);

/**
 * Query an InferenceModel output counts.
 */
enum TRACT_RESULT tract_inference_model_output_count(const struct TractInferenceModel *model,
                                                     uintptr_t *outputs);

/**
 * Query the name of a model input.
 *
 * The returned name must be freed by the caller using tract_free_cstring.
 */
enum TRACT_RESULT tract_inference_model_input_name(const struct TractInferenceModel *model,
                                                   uintptr_t input,
                                                   char **name);

/**
 * Query the name of a model output.
 *
 * The returned name must be freed by the caller using tract_free_cstring.
 */
enum TRACT_RESULT tract_inference_model_output_name(const struct TractInferenceModel *model,
                                                    uintptr_t output,
                                                    int8_t **name);

enum TRACT_RESULT tract_inference_model_input_fact(const struct TractInferenceModel *model,
                                                   uintptr_t input_id,
                                                   struct TractInferenceFact **fact);

/**
 * Set an input fact of an InferenceModel.
 *
 * The `fact` argument is only borrowed by this function, it still must be destroyed.
 * `fact` can be set to NULL to erase the current output fact of the model.
 */
enum TRACT_RESULT tract_inference_model_set_input_fact(struct TractInferenceModel *model,
                                                       uintptr_t input_id,
                                                       const struct TractInferenceFact *fact);

/**
 * Change the model outputs nodes (by name).
 *
 * `names` is an array containing `len` pointers to null terminated strings.
 */
enum TRACT_RESULT tract_inference_model_set_output_names(struct TractInferenceModel *model,
                                                         uintptr_t len,
                                                         const char *const *names);

/**
 * Query an output fact for an InferenceModel.
 *
 * The return model must be freed using `tract_inference_fact_destroy`.
 */
enum TRACT_RESULT tract_inference_model_output_fact(const struct TractInferenceModel *model,
                                                    uintptr_t output_id,
                                                    struct TractInferenceFact **fact);

/**
 * Set an output fact of an InferenceModel.
 *
 * The `fact` argument is only borrowed by this function, it still must be destroyed.
 * `fact` can be set to NULL to erase the current output fact of the model.
 */
enum TRACT_RESULT tract_inference_model_set_output_fact(struct TractInferenceModel *model,
                                                        uintptr_t output_id,
                                                        const struct TractInferenceFact *fact);

/**
 * Analyse an InferencedModel in-place.
 */
enum TRACT_RESULT tract_inference_model_analyse(struct TractInferenceModel *model);

/**
 * Convenience function to obtain an optimized TypedModel from an InferenceModel.
 *
 * This function takes ownership of the InferenceModel `model` whether it succeeds
 * or not. `tract_inference_model_destroy` must not be used on `model`.
 *
 * On the other hand, caller will be owning the newly created `optimized` model.
 */
enum TRACT_RESULT tract_inference_model_into_optimized(struct TractInferenceModel **model,
                                                       struct TractModel **optimized);

/**
 * Transform a fully analysed InferenceModel to a TypedModel.
 *
 * This function takes ownership of the InferenceModel `model` whether it succeeds
 * or not. `tract_inference_model_destroy` must not be used on `model`.
 *
 * On the other hand, caller will be owning the newly created `optimized` model.
 */
enum TRACT_RESULT tract_inference_model_into_typed(struct TractInferenceModel **model,
                                                   struct TractModel **typed);

/**
 * Destroy an InferenceModel.
 */
enum TRACT_RESULT tract_inference_model_destroy(struct TractInferenceModel **model);

/**
 * Query an InferenceModel input counts.
 */
enum TRACT_RESULT tract_model_input_count(const struct TractModel *model, uintptr_t *inputs);

/**
 * Query an InferenceModel output counts.
 */
enum TRACT_RESULT tract_model_output_count(const struct TractModel *model, uintptr_t *outputs);

/**
 * Query the name of a model input.
 *
 * The returned name must be freed by the caller using tract_free_cstring.
 */
enum TRACT_RESULT tract_model_input_name(const struct TractModel *model,
                                         uintptr_t input,
                                         char **name);

/**
 * Query the input fact of a model.
 *
 * Thre returned fact must be freed with tract_fact_destroy.
 */
enum TRACT_RESULT tract_model_input_fact(const struct TractModel *model,
                                         uintptr_t input_id,
                                         struct TractFact **fact);

/**
 * Query the name of a model output.
 *
 * The returned name must be freed by the caller using tract_free_cstring.
 */
enum TRACT_RESULT tract_model_output_name(const struct TractModel *model,
                                          uintptr_t output,
                                          char **name);

/**
 * Query the output fact of a model.
 *
 * Thre returned fact must be freed with tract_fact_destroy.
 */
enum TRACT_RESULT tract_model_output_fact(const struct TractModel *model,
                                          uintptr_t input_id,
                                          struct TractFact **fact);

/**
 * Change the model outputs nodes (by name).
 *
 * `names` is an array containing `len` pointers to null terminated strings.
 */
enum TRACT_RESULT tract_model_set_output_names(struct TractModel *model,
                                               uintptr_t len,
                                               const char *const *names);

/**
 * Give value one or more symbols used in the model.
 *
 * * symbols is an array of `nb_symbols` pointers to null-terminated UTF-8 string for the symbols
 *   names to substitue
 * * values is an array of `nb_symbols` integer values
 */
enum TRACT_RESULT tract_model_concretize_symbols(struct TractModel *model,
                                                 uintptr_t nb_symbols,
                                                 const int8_t *const *symbols,
                                                 const int64_t *values);

/**
 * Pulsify the model
 *
 * * stream_symbol is the name of the stream symbol
 * * pulse expression is a dim to use as the pulse size (like "8", "P" or "3*p").
 */
enum TRACT_RESULT tract_model_pulse_simple(struct TractModel **model,
                                           const int8_t *stream_symbol,
                                           const int8_t *pulse_expr);

/**
 * Apply a transform to the model.
 */
enum TRACT_RESULT tract_model_transform(struct TractModel *model, const int8_t *transform);

/**
 * Declutter a TypedModel in-place.
 */
enum TRACT_RESULT tract_model_declutter(struct TractModel *model);

/**
 * Optimize a TypedModel in-place.
 */
enum TRACT_RESULT tract_model_optimize(struct TractModel *model);

/**
 * Perform a profile of the model using the provided inputs.
 */
enum TRACT_RESULT tract_model_profile_json(struct TractModel *model,
                                           struct TractValue **inputs,
                                           int8_t **json);

/**
 * Convert a TypedModel into a TypedRunnableModel.
 *
 * This function transfers ownership of the `model` argument to the newly-created `runnable` model.
 *
 * Runnable are reference counted. When done, it should be released with `tract_runnable_release`.
 */
enum TRACT_RESULT tract_model_into_runnable(struct TractModel **model,
                                            struct TractRunnable **runnable);

/**
 * Query the number of properties in a model.
 */
enum TRACT_RESULT tract_model_property_count(const struct TractModel *model, uintptr_t *count);

/**
 * Query the properties names of a model.
 *
 * The "names" array should be big enough to fit `tract_model_property_count` string pointers.
 *
 * Each name will have to be freed using `tract_free_cstring`.
 */
enum TRACT_RESULT tract_model_property_names(const struct TractModel *model, int8_t **names);

/**
 * Query a property value in a model.
 */
enum TRACT_RESULT tract_model_property(const struct TractModel *model,
                                       const int8_t *name,
                                       struct TractValue **value);

/**
 * Destroy a TypedModel.
 */
enum TRACT_RESULT tract_model_destroy(struct TractModel **model);

/**
 * Spawn a session state from a runnable model.
 *
 * This function does not take ownership of the `runnable` object, it can be used again to spawn
 * other state instances. The runnable object is internally reference counted, it will be
 * kept alive as long as any associated `State` exists (or as long as the `runnable` is not
 * explicitely release with `tract_runnable_release`).
 *
 * `state` is a newly-created object. It should ultimately be detroyed with `tract_state_destroy`.
 */
enum TRACT_RESULT tract_runnable_spawn_state(struct TractRunnable *runnable,
                                             struct TractState **state);

/**
 * Convenience function to run a stateless model.
 *
 * `inputs` is a pointer to an pre-existing array of input TractValue. Its length *must* be equal
 * to the number of inputs of the models. The function does not take ownership of the input
 * values.
 * `outputs` is a pointer to a pre-existing array of TractValue pointers that will be overwritten
 * with pointers to outputs values. These values are under the responsiblity of the caller, it
 * will have to release them with `tract_value_destroy`.
 */
enum TRACT_RESULT tract_runnable_run(struct TractRunnable *runnable,
                                     struct TractValue **inputs,
                                     struct TractValue **outputs);

/**
 * Query a Runnable input counts.
 */
enum TRACT_RESULT tract_runnable_input_count(const struct TractRunnable *model, uintptr_t *inputs);

/**
 * Query an Runnable output counts.
 */
enum TRACT_RESULT tract_runnable_output_count(const struct TractRunnable *model,
                                              uintptr_t *outputs);

enum TRACT_RESULT tract_runnable_release(struct TractRunnable **runnable);

/**
 * Create a TractValue (aka tensor) from caller data and metadata.
 *
 * This call copies the data into tract space. All the pointers only need to be alive for the
 * duration of the call.
 *
 * rank is the number of dimensions of the tensor (i.e. the length of the shape vector).
 *
 * The returned value must be destroyed by `tract_value_destroy`.
 */
enum TRACT_RESULT tract_value_from_bytes(DatumType datum_type,
                                         uintptr_t rank,
                                         const uintptr_t *shape,
                                         void *data,
                                         struct TractValue **value);

/**
 * Destroy a value.
 */
enum TRACT_RESULT tract_value_destroy(struct TractValue **value);

/**
 * Inspect part of a value. Except `value`, all argument pointers can be null if only some specific bits
 * are required.
 */
enum TRACT_RESULT tract_value_as_bytes(struct TractValue *value,
                                       DatumType *datum_type,
                                       uintptr_t *rank,
                                       const uintptr_t **shape,
                                       const void **data);

/**
 * Run a turn on a model state
 *
 * `inputs` is a pointer to an pre-existing array of input TractValue. Its length *must* be equal
 * to the number of inputs of the models. The function does not take ownership of the input
 * values.
 * `outputs` is a pointer to a pre-existing array of TractValue pointers that will be overwritten
 * with pointers to outputs values. These values are under the responsiblity of the caller, it
 * will have to release them with `tract_value_destroy`.
 */
enum TRACT_RESULT tract_state_run(struct TractState *state,
                                  struct TractValue **inputs,
                                  struct TractValue **outputs);

/**
 * Query a State input counts.
 */
enum TRACT_RESULT tract_state_input_count(const struct TractState *state, uintptr_t *inputs);

/**
 * Query an State output counts.
 */
enum TRACT_RESULT tract_state_output_count(const struct TractState *state, uintptr_t *outputs);

enum TRACT_RESULT tract_state_destroy(struct TractState **state);

/**
 * Parse a fact specification string into an Fact.
 *
 * The returned fact must be free with `tract_fact_destroy`.
 */
enum TRACT_RESULT tract_fact_parse(struct TractModel *model,
                                   const char *spec,
                                   struct TractFact **fact);

/**
 * Write a fact as its specification string.
 *
 * The returned string must be freed by the caller using tract_free_cstring.
 */
enum TRACT_RESULT tract_fact_dump(const struct TractFact *fact, char **spec);

enum TRACT_RESULT tract_fact_destroy(struct TractFact **fact);

/**
 * Parse a fact specification string into an InferenceFact.
 *
 * The returned fact must be free with `tract_inference_fact_destroy`.
 */
enum TRACT_RESULT tract_inference_fact_parse(struct TractInferenceModel *model,
                                             const char *spec,
                                             struct TractInferenceFact **fact);

/**
 * Creates an empty inference fact.
 *
 * The returned fact must be freed by the caller using tract_inference_fact_destroy
 */
enum TRACT_RESULT tract_inference_fact_empty(struct TractInferenceFact **fact);

/**
 * Write an inference fact as its specification string.
 *
 * The returned string must be freed by the caller using tract_free_cstring.
 */
enum TRACT_RESULT tract_inference_fact_dump(const struct TractInferenceFact *fact, char **spec);

/**
 * Destroy a fact.
 */
enum TRACT_RESULT tract_inference_fact_destroy(struct TractInferenceFact **fact);
