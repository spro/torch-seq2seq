# torch-seq2seq

Uses a word-level Seq2Seq network in Torch to learn to translate the synthetic language *Espançanese*:

```
> te grillo wa silencieusement matte iru
= your cricket is sleeping silently
```

The language has a limited vocabulary, but is useful to demonstrate long and short term dependencies in grammar, including:

* subject-object-verb order
* negative particles
* flipped adjective and adverb orders
* gendered particles and pronouns
* surrounding punctuation

## Model

![](https://i.imgur.com/NzvlG3X.png)

The encoder turns a sequence of words into a vector of size `hidden_size` with a linear layer and LSTM layer:

```lua
encoder_lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)

encoder = nn.Sequential()
    :add(nn.OneHot(n_tokens_in))
    :add(nn.Linear(n_tokens_in, opt.hidden_size))
    :add(encoder_lstm)

encoder = nn.Sequencer(encoder)
```

The decoder takes a word or start marker and outputs the likeliness of the next word or end marker:

```lua
decoder_lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)

decoder = nn.Sequential()
    :add(nn.OneHot(n_tokens_out + 2))
    :add(nn.Linear(n_tokens_out + 2, opt.hidden_size))
    :add(decoder_lstm)
    :add(nn.Linear(opt.hidden_size, n_tokens_out + 2))
    :add(nn.LogSoftMax())

decoder = nn.Sequencer(decoder)
```

### Encoder / decoder coupling

The encoder creates a "context vector" but the decoder doesn't have explicit inputs for this vector. Instead the hidden state is copied directly from encoder to decoder, as demonstrated in [rnn's seq2seq example](https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua).

When sampling, the last output of the encoder LSTM is copied as the previous output of the decoder LSTM:

```lua
decoder_lstm.userPrevOutput = nn.rnn.recursiveCopy(
    decoder_lstm.userPrevOutput,
    encoder_lstm.outputs[last_index]
)
decoder_lstm.userPrevCell = nn.rnn.recursiveCopy(
    decoder_lstm.userPrevCell,
    encoder_lstm.cells[last_index]
)
```

After sampling, gradients are copied back from the decoder to the encoder:

```lua
encoder_lstm.userNextGradCell = nn.rnn.recursiveCopy(
    encoder_lstm.userNextGradCell,
    decoder_lstm.userGradPrevCell
)
encoder_lstm.gradPrevOutput = nn.rnn.recursiveCopy(
    encoder_lstm.gradPrevOutput,
    decoder_lstm.userGradPrevOutput
)
```


## Training

```
$ th train.lua -hidden_size 100 -learning_rate 0.1

-hidden_size         Hidden size of LSTM layer [100]
-learning_rate       Learning rate [0.1]
-learning_rate_decay Learning rate decay [1e-05]
-max_length          Maximum output length [20]
-n_epochs            Number of epochs to train [100000]
```

Every several iterations it will sample a new set of sentences and print them as:

```
> input
= target
~ output (sampled)
```

It starts out with senseless guessing:

```
> watashi wa matte iru
= i am sleeping
~ kick cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat

> ? la ensalada verde wa doko desu ka ?
= where is the green salad ?
~ is why why definitely definitely definitely definitely definitely definitely 

> ? ma computadora noir wa doko desu ka ?
= where is my black computer ?
~ i will not not the shoe
```

But soon learns general sentence structure:

```
> la medusa wo yaku
= i will grill the jellyfish
~ i will eat the goose

> la puerta sal wo daku sen
= i will not hug the dirty door
~ i will not hug the sad box

> te zapato et te pajaro wo kiru
= i will cut your shoe and your bird
~ i will cut your hedgehog and your boat
```

And learns most of the vocabulary by the end of training:

```
> ! te jardin noveau wo nagusameru sen !
= i definitely will not comfort your new garden
~ i definitely will not comfort your new garden

> ? dare ga me erizo petit o mamotte iru nodesu ka ?
= who is protecting my small hedgehog ?
~ who is protecting my small hedgehog ?

> la medusa petit wa el grillo o poliment yaita
= the small jellyfish grilled the cricket politely
~ the small jellyfish grilled the cricket politely
```
