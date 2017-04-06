print('--------------------------------------------------------------------------------')
require 'nn'
require 'rnn'
require 'dpnn'
require 'optim'
display = require 'display'

-- Parse command line arguments

cmd = torch.CmdLine()
cmd:text()

cmd:option('-hidden_size', 100, 'Hidden size of LSTM layer')
cmd:option('-learning_rate', 0.1, 'Learning rate')
cmd:option('-learning_rate_decay', 1e-7, 'Learning rate decay')
cmd:option('-max_length', 20, 'Maximum output length')
cmd:option('-n_epochs', 100000, 'Number of epochs to train')

opt = cmd:parse(arg)

require 'data'

-- Build the model

encoder_lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)

encoder = nn.Sequential()
    :add(nn.OneHot(n_words_in))
    :add(nn.Linear(n_words_in, opt.hidden_size))
    :add(encoder_lstm)

encoder = nn.Sequencer(encoder)

decoder_lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)

decoder = nn.Sequential()
    :add(nn.OneHot(n_words_out + 2))
    :add(nn.Linear(n_words_out + 2, opt.hidden_size))
    :add(decoder_lstm)
    :add(nn.Linear(opt.hidden_size, n_words_out + 2))
    :add(nn.LogSoftMax())

decoder = nn.Sequencer(decoder)

SOS = n_words_out + 1
EOS = n_words_out + 2

decoder:remember()

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- Helper functions

seen_inputs = {}

function makeEncoderInputs(tokens)
    local inputs = {}
    local joined = table.concat(tokens, ' ')
    local had_seen = seen_inputs[joined]
    seen_inputs[joined] = true
    for ci = 1, #tokens do
        table.insert(inputs, wordToIn(tokens[ci]))
    end
    return torch.LongTensor(inputs):view(-1, 1), not had_seen
end

function makeDecoderInput(word)
    return torch.LongTensor({word}):view(-1, 1)
end

function makeDecoderInputs(tokens)
    local word_inputs = {}
    table.insert(word_inputs, SOS)
    for ci = 1, #tokens do
        local word = tokens[ci]
        table.insert(word_inputs, wordToOut(word))
    end
    return torch.LongTensor(word_inputs):view(-1, 1)
end

function makeTargetOutputs(tokens)
    local inputs = {}
    for ci = 1, #tokens do
        table.insert(inputs, wordToOut(tokens[ci]))
    end
    table.insert(inputs, EOS)
    return torch.LongTensor(inputs):view(-1, 1)
end

-- Sampling
--------------------------------------------------------------------------------

function sample()
    encoder:forget()
    decoder:forget()

    local sentence = makeSentence()
    local input_tokens = tokenize(sentence[1])

    local encoder_inputs, is_new = makeEncoderInputs(input_tokens)
    local encoder_outputs = encoder:forward(encoder_inputs)
    local last_index = #encoder_lstm.outputs

    -- Last layer of encoder to first layer of decoder
    decoder_lstm.userPrevOutput = nn.rnn.recursiveCopy(decoder_lstm.userPrevOutput, encoder_lstm.outputs[last_index])
    decoder_lstm.userPrevCell = nn.rnn.recursiveCopy(decoder_lstm.userPrevCell, encoder_lstm.cells[last_index])

    -- Start with start marker
    local decoder_inputs = makeDecoderInput(SOS)
    local sampled = ''

    for i = 1, opt.max_length do
        local decoder_output = decoder:forward(decoder_inputs)

        -- Get most likely output
        local max_score, max_val = decoder_output:view(-1):max(1)
        max_val = max_val[1]

        if max_val == EOS then
            break
        else
            -- Next input is this output
            decoder_inputs = makeDecoderInput(max_val)
            sampled = sampled .. outToWord(max_val) .. ' '
        end
    end

    print(string.format('\n> %s\n= %s\n~ %s', sentence[1], sentence[2], sampled))
    return sampled
end

-- Training
--------------------------------------------------------------------------------

-- Run a loop of optimization

n_epoch = 1

function train()
	encoder:zeroGradParameters()
	decoder:zeroGradParameters()
	encoder:forget()
	decoder:forget()

    -- Inputs and targets

	local sentence = makeSentence()
    local input_tokens = tokenize(sentence[1])
    local output_tokens = tokenize(sentence[2])

    local encoder_inputs = makeEncoderInputs(input_tokens)
    local decoder_inputs = makeDecoderInputs(output_tokens)
    local target_outputs = makeTargetOutputs(output_tokens)

	-- Forward pass

	local encoder_outputs = encoder:forward(encoder_inputs)
    local last_index = #encoder_lstm.outputs
    
    decoder_lstm.userPrevOutput = nn.rnn.recursiveCopy(decoder_lstm.userPrevOutput, encoder_lstm.outputs[last_index])
    decoder_lstm.userPrevCell = nn.rnn.recursiveCopy(decoder_lstm.userPrevCell, encoder_lstm.cells[last_index])

	local decoder_outputs = decoder:forward(decoder_inputs)
	local err = criterion:forward(decoder_outputs, target_outputs)

	-- Backward pass

	local decoder_grad_outputs = criterion:backward(decoder_outputs, target_outputs)
	decoder:backward(decoder_inputs, decoder_grad_outputs)

    encoder_lstm:setGradHiddenState(last_index, decoder_lstm:getGradHiddenState(0))

	local zero_tensor = torch.Tensor(encoder_outputs):zero()
	encoder:backward(encoder_inputs, zero_tensor)

	decoder:updateParameters(opt.learning_rate)
	encoder:updateParameters(opt.learning_rate)

	return err
end

err = 0
errs = {}
learning_rates = {}
plot_every = 10
sample_every = 500

for n_epoch = 1, opt.n_epochs do
	err = err + train()
    opt.learning_rate = opt.learning_rate * (1 - opt.learning_rate_decay)

    -- Plot every plot_every
    if n_epoch % plot_every == 0 then
        err = err / 10
        if err > 0 and err < 999 then
            table.insert(errs, {n_epoch, err})
            display.plot(errs, {win='dec errs'})
            table.insert(learning_rates, {n_epoch, opt.learning_rate})
            display.plot(learning_rates, {win='dec learning_rates'})
        end
        err = 0
    end

    -- Sample every sample_every
    if n_epoch % sample_every == 0 then
        sample()
    end
end

torch.save('encoder1.t7', encoder)
torch.save('decoder1.t7', decoder)
