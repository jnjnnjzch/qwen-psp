#include "qwen_model.h"

static int append_text(const Tokenizer *tokenizer, const char *text,
                       int *output, int count, int capacity) {
    int produced = tokenizer_encode(tokenizer, text, output + count, capacity - count);
    return produced < 0 ? -1 : count + produced;
}

/* 聊天模板将用户文本放入模型训练时使用的消息格式中。
 * user、assistant 表示消息角色，两个 im 标记表示消息起止。 */
int tokenizer_build_chat_prompt(const QwenModel *model, const char *text, int *output, int capacity) {
    int count = 0;
    if (!model || !text || !output || capacity < 4) return -1;
    output[count++] = model->config.im_start_id;
    count = append_text(&model->tokenizer, "user\n", output, count, capacity);
    if (count < 0) return -1;
    count = append_text(&model->tokenizer, text, output, count, capacity);
    if (count < 0 || count >= capacity) return -1;
    output[count++] = model->config.im_end_id;
    count = append_text(&model->tokenizer, "\n", output, count, capacity);
    if (count < 0 || count >= capacity) return -1;
    output[count++] = model->config.im_start_id;
    return append_text(&model->tokenizer, QWEN_ASSISTANT_PREFIX, output, count, capacity);
}
