label = get_label(1);

answer = get_answer(1);

loss_adam = []; 

total_time = 0; % 누적 시간 초기화




function [loss, grad] = compute_loss_and_gradient(sigma, label)
    % 예측값 계산
    prediction = fun(sigma);

    % MSE 계산
    loss = sum((prediction - label).^2) / 18;

    % 그래디언트 계산
    grad = compute_gradient(sigma, label);
end

function mae = compute_MAE(answer, sigma)
    difference = abs(answer - sigma);
    mae = mean(difference)*1000;
end

function grad = compute_gradient(sigma, label)


    % 그래디언트 계산
    grad = zeros(size(sigma));
    epsilon = 1e-6; % epsilon 값
    for i = 1:length(sigma)
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

% 하이퍼파라미터 설정
learning_rate = 0.1; % 초기 학습률
max_iterations = 100; % 최대 반복 횟수
beta1 = 0.9; % Adam 알고리즘 하이퍼파라미터
beta2 = 0.999; % Adam 알고리즘 하이퍼파라미터
epsilon = 1e-8; % 안정성을 위한 작은 값




sigma_list = [];
loss_adam = [];
mae_adam = [];
iteration_adam = [];

for data_num = 1:20
    label = get_label(data_num);

    answer = get_answer(data_num);

    accumulated_grad = zeros(size(sigma));
    
    sigma = ones(11,1)*0.2;

    
    % Adam 알고리즘을 이용한 최적화
    m = zeros(size(sigma)); % 1차 모멘텀 벡터
    v = zeros(size(sigma)); % 2차 모멘텀 벡터
    for iter = 1:max_iterations
        

    
        % 손실과 그래디언트 계산
        [loss, grad] = compute_loss_and_gradient(sigma, label);
    
        % Adam 업데이트
        m = beta1 * m + (1 - beta1) * grad;
        v = beta2 * v + (1 - beta2) * grad.^2;
        m_hat = m / (1 - beta1^iter);
        v_hat = v / (1 - beta2^iter);
        sigma = sigma - learning_rate * m_hat ./ (sqrt(v_hat) + epsilon);
    

        
        mae = compute_MAE(answer, sigma);
        if mod(iter, 10) == 0
            disp(['Iteration: ', num2str(iter), ', Loss: ', num2str(loss)]);
    
            disp(['loss with answer: ', num2str(mae)]);
        end
        elapsed_time = toc;
        total_time = total_time + elapsed_time;
        
        if mod(iter, 1) == 0
            loss_adam(data_num, iter) = [loss];
            mae_adam(data_num, iter) = [mae];
            iteration_adam(data_num, iter) = [iter];
        end
    
    
    end
    sigma_list(:,data_num) = sigma;


end

disp(sigma_list)
%plot(iteration_adam, loss_list_adam, '-o'); % x축을 누적 시간, y축을 함수 값으로 하는 그래프 그리기
%plot(iteration_adam, mae_adam, '-o');
%xlabel('iteration 횟수'); % x축 라벨 설정
%ylabel('MAE'); % y축 라벨 설정
%title('iteration에 따른 MAE 변화'); % 그래프 제목 설정