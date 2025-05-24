import random



def count_chinese_characters(line):
    chinese_count = 0
    for char in line:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
    contains_chinese = chinese_count > 0
    return chinese_count


def mixed_err_chinese_remove(line):
    words = line.split()
    chinese_words = 0
    for word in words:
        if count_chinese_characters(word) > 0:
            if count_chinese_characters(word)/len(word) < 0.5 and len(word) > 3: 
                return True
            chinese_words += 1
    
    if chinese_words/len(words) > 0.4:
        return True
    return False

def remove_chinese(messages):
    flag = False 
    for message in messages:
        if 'content' in message:
            content = message['content']
            if count_chinese_characters(content) > 0:
                flag = True
                break
    return flag

def prune_chinese(data):

    count_prune = 0
    keep = []
    for message in data:
        should_remove = remove_chinese(message['messages'])
        if should_remove:
            count_prune += 1
        else:
            keep.append(message)
    print(f"Removed {count_prune} messages with Chinese characters.")
    return keep





def news_sys_prompt(data):

    for message in data:
        if 'messages' in message and message['messages'][0]['role'] != 'system':
            msg = [
                {
                    'role': 'system',
                    'content': 'Bạn là trợ lý hữu ích. /no_think'
                }
            ]
            msg.extend(message['messages'])
            message['messages'] = msg

    return data



def process_thinking(object):

    question = object['question']
    answer = object['answer']
    think = object['think']

    pair = 'en_en'

    if 'vi_question' in object and 'vi_answer' in object:
        # Vietnamese
        vi_question = object['vi_question']
        vi_answer = object['vi_answer']
        

        if count_chinese_characters(question) == 0 and count_chinese_characters(answer) == 0:
            # Both question and answer contain Chinese characters
            if 'vi_thinking' in object and count_chinese_characters(object['vi_thinking']) > 0:
                # Thinking contains Chinese characters
                sys_prompts = ['Bạn là trợ lí có khả năng suy nghĩ. Hãy suy nghĩ kĩ và trả lời câu hỏi của người dùng. Đặt phần suy nghĩ của bạn trong tag <think></think> /think', 'Bạn là một trợ lý thông minh.  /think', 'bạn là một trợ lý thông minh. /vietnamese /think']

                # sys_prompt = random.choice(sys_prompts)
                think = object['vi_thinking']

                pair = 'vi_vi'

            else:
                sys_prompts = ["You are a reasoning assistant. You must think about the reasoning process in mind then provide user with answer. The reasoning process are enclosed in <think> </think> /english /think",  "You are a smart assistant /english /think"]

                # sys_prompt = random.choice(sys_prompts)

                think = "The translation of the user request is:\n\n" + question + "\n\n" + think

                pair = 'vi_en'

            question = vi_question
            answer = vi_answer

        else:
            return None, False, None

    else:
        sys_prompts = ["You are a reasoning assistant. You must think about the reasoning process in mind then provide user with answer. The reasoning process are enclosed in <think> </think> /think", "You are a smart assistant /think", "You are a smart assistant /think"]

    messages = [
        {
            'role': 'system',
            'content': random.choice(sys_prompts)
        },
        {
            'role': 'user',
            'content': question
        },
        {
            'role': 'assistant',
            'content': "<think>\n" + think + "\n</think>\n\n" + answer
        }
    ]


    if random.random() < 0.5:
        messages[1]['content'] = messages[1]['content'] + " /think"

    return pair, True, messages



def process_messages(messages, template):

    print('process template', template)
    
    return_messages = []
    
    if template == 'qwen3':
        if 'thinking' in messages[0]:
            for message in messages:
                pair, flag, message = process_thinking(message)
                if flag:
                    return_messages.append(message)
        else:
            return_messages = news_sys_prompt(messages)

    elif template == 'r1':
        if 'thinking' in messages[0]:
            for message in messages:
                pair, flag, message = process_thinking(message)
                if flag:
                    return_messages.append(message)
        else:
            return messages 
    
    else:
        return_messages = messages


    return return_messages