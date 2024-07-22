

function [loss, grad] = compute_loss_and_gradient(sigma, label)

    prediction = fun(sigma);

    loss = sum((prediction - label).^2) / 18;

    grad = compute_gradient(sigma, label);
end

function mae = compute_MAE(answer, sigma)
    difference = abs(answer - sigma);
    mae = mean(difference)*1000;
end

function grad = compute_gradient(sigma, label)

    % calculate gradient
    grad = zeros(size(sigma));
    epsilon = 1e-6; % epsilon 값
    for i = 1:length(sigma)
        % 중심 차분을 사용하여 그래디언트 근사화
        sigma_plus_delta = sigma;
        sigma_plus_delta(i) = sigma_plus_delta(i) + epsilon;
        sigma_minus_delta = sigma;
        sigma_minus_delta(i) = sigma_minus_delta(i) - epsilon;

        prediction_plus_delta = fun(sigma_plus_delta);
        prediction_minus_delta = fun(sigma_minus_delta);

        loss_plus_delta = sum((prediction_plus_delta - label).^2) / 18;
        loss_minus_delta = sum((prediction_minus_delta - label).^2) / 18;

        grad(i) = (loss_plus_delta - loss_minus_delta) / (2 * epsilon);
    end
end



% Set hyperparameters
learning_rate = 0.1;
max_iterations = 100; 


sigma = ones(11,1)*0.2;

loss = sum(abs(fun(sigma) - label)) / 18;

loss_adagrad = [loss];



iteration_adagrad = [0]; % 누적 시간 저장 리스트
mae_adagrad = [compute_MAE(sigma, answer)];



disp(['Iteration: ', num2str(0), ', Loss: ', num2str(loss)]);

% AdaGrad optimization
epsilon = 1e-8; 


accumulated_grad = zeros(size(sigma));

sigma_list_adagrad = [];
loss_adagrad = [];
mae_adagrad = [];
iteration_adagrad = [];

for data_num = 1:20
    label = get_label(data_num);

    answer = get_answer(data_num);
    
    
    sigma = ones(11,1)*0.2;


    for iter = 1:max_iterations
    
        % calculate loss and gradient
        [loss, grad] = compute_loss_and_gradient(sigma, label);
    
        % AdaGrad 업데이트
        accumulated_grad = accumulated_grad + grad.^2;
        sigma = sigma - learning_rate * grad ./ (sqrt(accumulated_grad) + epsilon);
        
        mae = compute_MAE(answer, sigma);
    
    
        if mod(iter, 1) == 0
            iteration_adagrad(data_num, iter) = [iter];
            mae_adagrad(data_num, iter) = [mae];
            loss_adagrad(data_num, iter) = [loss];
        end
        if mod(iter, 10) == 0
            disp(['Iteration: ', num2str(iter), ', Loss: ', num2str(loss)]);
    
            disp(['loss with answer: ', num2str(mae)]);
        end
   
    end
    sigma_list_adagrad(:,data_num) = sigma;
end

disp(['loss: ', num2str(loss)]);
disp(sigma_list_adagrad);
