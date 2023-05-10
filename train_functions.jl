using Flux

function train_model!(selected_model, train_data, test_data ; epoch_lim = 20000)
    model_loss_history = Float32[1]
    test_loss_history = Float32[1]
    x_train = train_data[:, [1]]'
    y_train = train_data[:, [2]]'
    x_test = test_data[:, [1]]'
    y_test = test_data[:, [2]]'

    loss1(model, x, y) = Flux.mse(model(x), y)
    opt_state = Flux.setup(Flux.Adam(), selected_model)

    epoch = 0 
    while epoch < epoch_lim && loss_history[end] > 0.0001
        epoch += 1 
        global selected_model, state_tree
        dLdm, _, _ = gradient(loss1, selected_model, x_train, y_train)
        # state_tree, selected_model = Optimisers.update(state_tree, selected_model, dLdm)
        Flux.update!(opt_state, selected_model, dLdm)

        training_loss = loss1(selected_model, x_train, y_train)
        testing_loss  = loss1(selected_model, x_test , y_test)
        push!(model_loss_history, training_loss)
        push!(test_loss_history, testing_loss)
        if epoch % 50 == 0
            println("Epoch: $epoch, Loss: $model_loss_history[end], Test Loss: $testing_loss")
        end
        
    end
    println("====="^50)
    println("DONE")
    training_loss = loss1(selected_model, x_train, y_train)
    testing_loss  = loss1(selected_model, x_test , y_test)
    println("Epoch: $epoch, Loss: $training_loss, Test Loss: $testing_loss")

    return selected_model, model_loss_history, test_loss_history
end