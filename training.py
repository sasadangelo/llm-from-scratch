import torch
from vocabulary import Vocabulary
from simple_tokenizer import SimpleTokenizer
from simple_dataloader import SimpleDataLoader

if __name__ == "__main__":
    vocabulary = Vocabulary()
    vocabulary.add_book("robinson-crusoe.txt")
    tokenizer = SimpleTokenizer(vocabulary)
    print("Vocabulary length: ", vocabulary.len())
    # ids1 = tokenizer.encode("Who is Robinson Crusoe?")
    # print(ids1)
    # ids2 = tokenizer.encode("How did he get to the desert island?")
    # print(ids2)
    # ids3 = tokenizer.encode("Who is the author of this novel?")
    # print(ids3)
    # ids4 = tokenizer.encode("Who is Friday in the story?")
    # print(ids4)
    # question1 = tokenizer.decode(ids1)
    # print("Decode question 1: ", question1)
    # question2 = tokenizer.decode(ids2)
    # print("Decode question 2: ", question2)
    # question3 = tokenizer.decode(ids3)
    # print("Decode question 3: ", question3)
    # question4 = tokenizer.decode(ids4)
    # print("Decode question 4: ", question4)
    with open("robinson-crusoe.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    #enc_text = tokenizer.encode(raw_text)
    #print(len(enc_text))
    #enc_sample = enc_text[50:]
    #print(enc_sample)
    #context_size = 4         #1
    #x = enc_sample[:context_size]
    #y = enc_sample[1:context_size+1]
    #print(f"x: {x}")
    #print(f"y:      {y}")
    #for i in range(1, context_size+1):
    #    context = enc_sample[:i]
    #    desired = enc_sample[i]
    #    print(context, "---->", desired)
    #for i in range(1, context_size+1):
    #    context = enc_sample[:i]
    #    desired = enc_sample[i]
    #    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

    max_length=4
    dataloader = SimpleDataLoader(
        raw_text, tokenizer, batch_size=8, max_length=max_length, stride=4, shuffle=False)
    data_iter = iter(dataloader)      #1
    inputs, target = next(data_iter)
    print("Token IDs: ", inputs)
    print("Input shape: ", inputs.shape)
    #second_batch = next(data_iter)
    #print(second_batch)

    #vocab_size = vocabulary.len()
    output_dim = 256
    #torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(vocabulary.len(), output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings)
    #print(embedding_layer.weight)
    #print(embedding_layer(torch.tensor([3])))
    #input_ids = torch.tensor([2, 3, 5, 1])
    #print(embedding_layer(input_ids))

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("Positional: ", pos_embeddings)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings)