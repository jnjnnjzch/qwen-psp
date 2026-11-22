#include "qwen.h"

#include <stdlib.h>

/* 一次生成分为两部分：处理用户提示的全部 token，然后反复生成新 token。
 * 模型 forward 的输入和输出都是数字；tokenizer 负责与文本之间的转换。 */
int qwen_generate_turn(QwenModel *model, QwenRuntime *runtime, Sampler *sampler,
                        const int *prompt, int prompt_count, int start_position,
                        int max_new_tokens, QwenTokenCallback callback, void *user,
                        int *generated_count, int *next_position, int *pending_token) {
    const QwenConfig *config = &model->config;
    int *history;
    int history_count = 0, index, token, generated = 0;
    int position = start_position, pending = -1;
    float *logits = NULL;
    if (generated_count) *generated_count = 0;
    if (pending_token) *pending_token = -1;
    if (prompt_count <= 0 || max_new_tokens < 0 || start_position < 0 ||
        start_position + prompt_count + max_new_tokens > config->context) return 0;
    history = malloc((size_t)(prompt_count + max_new_tokens) * sizeof(*history));
    if (!history) return 0;
    if (start_position == 0) qwen_runtime_reset(runtime, model);

    /* 处理提示，也称 prefill。最后一个提示 token 的 logits 用于选择第一个回答 token。 */
    for (index = 0; index < prompt_count; ++index) {
        history[history_count++] = prompt[index];
        logits = qwen_forward_token(model, runtime, prompt[index], position, index + 1 == prompt_count);
        ++position;
        if (!logits) { free(history); return 0; }
    }
    for (index = 0; index < max_new_tokens; ++index) {
        size_t length;
        const uint8_t *bytes;
        int keep_going;
        token = sampler_sample(sampler, logits, config->vocab_size, history, history_count);
        if (token == config->eos_id || token == config->im_end_id ||
            token == config->eom_id || token == config->eot_id) break;
        history[history_count++] = token;
        bytes = tokenizer_token_bytes(&model->tokenizer, token, &length);
        ++generated;
        keep_going = !callback || callback(token, bytes, length, user);
        if (!keep_going || index + 1 >= max_new_tokens) {
            /* 这个 token 已经显示出来，尚未送入 forward。
             * 继续会话时，将 pending 放在下一轮输入的最前面。 */
            pending = token;
            break;
        }
        logits = qwen_forward(model, runtime, token, position);
        ++position;
        if (!logits) { free(history); return 0; }
    }
    free(history);
    if (generated_count) *generated_count = generated;
    if (next_position) *next_position = position;
    if (pending_token) *pending_token = pending;
    return 1;
}

int qwen_generate(QwenModel *model, QwenRuntime *runtime, Sampler *sampler,
                   const int *prompt, int prompt_count, int max_new_tokens,
                   QwenTokenCallback callback, void *user, int *generated_count) {
    return qwen_generate_turn(model, runtime, sampler, prompt, prompt_count, 0,
                               max_new_tokens, callback, user, generated_count, NULL, NULL);
}
