import torch
import re
from transformers import BertForTokenClassification, BertTokenizerFast
from transformers import logging
logging.set_verbosity_warning()



pattern = r"[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\uff01]"
punctuation_list = re.findall(r'\\u[0-9a-fA-F]{4}', pattern)

punctuation_list = [bytes.fromhex(p[2:]).decode('utf-16be') for p in punctuation_list]

punctuation_map = {idx: value for idx, value in enumerate(punctuation_list)}
punctuation_map[13] = "word"




def tagging(isinstance):
    tag = []
    for c_idx in range(len(isinstance)):
        if(c_idx == len(isinstance) - 1):
            if isinstance[c_idx] in punctuation_list:
                continue
            else:
                tag.append(13)
        elif isinstance[c_idx] in punctuation_list:
            continue
        elif isinstance[c_idx + 1] in punctuation_list:
            tag.append(punctuation_list.index(isinstance[c_idx + 1]))
        else:
            tag.append(13) # 13代表word
    return tag


tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
label_all_tokens = True
def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels[word_idx])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels[word_idx] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids



class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            "bert-base-chinese", 
            num_labels=14)

    def forward(self, input_id, mask, label):
        output = self.bert(
            input_ids = input_id,
            attention_mask = mask, 
            labels=label, 
            return_dict=False)

        return output


def evaluate_one_text(model, sentence):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    

    if use_cuda:
        model = model.cuda()
    label = tagging(sentence)

    text = tokenizer(sentence, truncation=True, return_tensors="pt")
    


    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_label(sentence, label)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [punctuation_map[i] for i in predictions]
    print(f'original sentence\n{sentence}')

    print("recover_sentence")
    for idx in range(len(sentence)):
        if prediction_label[idx] == 'word':
            print(sentence[idx], end='')
        else: 
            print(sentence[idx], end='')
            print(prediction_label[idx], end='')

model = BertModel()
model.load_state_dict(torch.load("./model/50percent_model1.pth"))

if __name__ == "__main__":
    
    while True:
        sentence = input("请输入句子：")
        if sentence == "exit":
            break
        evaluate_one_text(model, sentence)
        print("\n")


#evaluate_one_text(model, "皇天后土实鉴此心背义忘恩天人共戮")