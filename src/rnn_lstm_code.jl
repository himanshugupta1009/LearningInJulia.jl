using Flux

rnn = Flux.RNNCell(2, 5)

x = rand(Float32, 2) # dummy data
h = rand(Float32, 5)  # initial hidden state

h, y = rnn(h, x)

m = Flux.Recur(rnn, h)
#=
r = RNN/LSTM command generates the entire recurrence as well!
No need to manually specify the recurrence.

However, if you use r = RNNCell/LSTMCell, you will have to manually specify the recurrence.
rnn = RNNCell(2, 5)
m = Flux.Recur(rnn,h)
=#


#BETTER WAY IS THIS!

using Flux

rnn = RNN(2, 5)  # or equivalently RNN(2 => 5)



#=
Q for Jackson

m = Chain(RNN(2 => 5), Dense(5 => 1))


So, the example in the doc is following!

function loss(x, y)
  sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end

seq_init = [rand(Float32, 2)]
seq_1 = [rand(Float32, 2) for i = 1:3]
seq_2 = [rand(Float32, 2) for i = 1:3]

y1 = [rand(Float32, 1) for i = 1:3]
y2 = [rand(Float32, 1) for i = 1:3]

X = [seq_1, seq_2]
Y = [y1, y2]
data = zip(X,Y)

Flux.reset!(m)
[m(x) for x in seq_init]

ps = Flux.params(m)
opt= Adam(1e-3)
Flux.train!(loss, ps, data, opt)


This works, but using the other way of Flux.train! doesn't work. WHY?

function loss2(model,x,y)
  sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end
seq_init = [rand(Float32, 2)]
seq_1 = [rand(Float32, 2) for i = 1:3]
seq_2 = [rand(Float32, 2) for i = 1:3]

y1 = [rand(Float32, 1) for i = 1:3]
y2 = [rand(Float32, 1) for i = 1:3]

X = [seq_1, seq_2]
Y = [y1, y2]
data = (X,Y)
Flux.reset!(m)
[m(x) for x in seq_init]

opt= Descent() or opt = Flux.setup(Adam(1e-3), model)
Flux.train!(loss2, m, data, opt)