% Inicializacija parametrov
% Definicija parametrov
T = 0.01; % Časovni korak, sampling
N = 400;    % Število samplov
N_APRBS = 3000;   % Št samplov za aprbs signal
amplitude_train = 2.5; % Amplituda za train
amplitude_test = 2.0;  % Amplituda za test signal
%Th = 0.20;
%padding = 100;
dV = 0.05;  % korak
c = zeros(2 / dV + 1, 1);  % zero vektor
stepSize = 0.05;
numSteps = N;
inputValues = 0:stepSize:2;  % Vrednosti vhodnega signala
outputValues = zeros(size(inputValues));  % Inicializacija vektorja za rezultate

% Glavna zanka za različne vrednosti vhodnega signala
for index = 1:length(inputValues)
    currentInput = inputValues(index);  % Trenutna vrednost vhodnega signala
    u = currentInput * ones(numSteps, 1);  % Vhodni signal (vektor enakih vrednosti)

    % Inicializacija stanja sistema
    state = [0, 0];

    % Inicializacija izhodnega vektorja
    outputSignal = zeros(numSteps, 1);

    % Simulacija sistema
    for step = 1:numSteps
        % Klic funkcije sistema - pridobivanje pozicije in hitrosti
        [position, velocity] = helicrane(u(step), state);

        % Posodobitev stanja
        state = [velocity, position];

        % Shranjevanje trenutnega izhoda
        outputSignal(step) = position;
    end

    % Izračun povprečja izhodnih vrednosti po prehodnem obdobju
    steadyStateOutput = mean(outputSignal(100:end));

    % Shranjevanje rezultata v outputValues
    outputValues(index) = steadyStateOutput;
end

% Risanje rezultata
figure;
plot(inputValues, outputValues);
title("Statična karakteristika sistema");
xlabel("Vhodni signal [V]");
ylabel("Izhodni signal [V]");


N = 4000;        % ŠT. aprbs samplov
amplitude = 2; % 2   % amplituda napetosti
Ts = 0.01;       % Sampling 
Th = 0.5;       % Hrizont spreminjanja aprbs signala
padding = 200;   % paddanje signala pred/po signalu

% generiranje aprbs signala
[u_train, t_train] = generateAPRBS(N, amplitude, Ts, Th, padding);
[u_test, t_test] = generateAPRBS(N, amplitude, Ts, Th, padding);

% simulacija odziva na aprbs
y_train = simulateHelicrane([0 0],u_train,t_train);
y_test = simulateHelicrane([0 0],u_test,t_test);


%u_train = u_train - 0.5;  % Now ranges approximately -0.5 to +0.5
%u_test = u_test - 0.5;

% Grafi vzbujalnega signala in odziva sistema na odziv

% Create figure and plot the signals and responses
figure('Position', [100, 100, 1400, 800]); 

% APRBS signal
subplot(2,2,1);
plot(t_train, u_train);
title('Trening APRBS Signal');
xlabel('Čas [s]');
ylabel('AMplituda');
grid on;

% Odziv sistema
subplot(2,2,2);
plot(t_train, y_train);
title('Odziv sistema');
xlabel('Čas [s]');
ylabel('Odziv');
grid on;

% Test APRBS
subplot(2,2,3);
plot(t_test, u_test);
title('Testni APRBS Signal');
xlabel('Čas [s]');
ylabel('Amplituda');
grid on;

% Odziv sistema na testni ARPBS
subplot(2,2,4);
plot(t_test, y_test);
title('Odziv sistema');
xlabel('Čas [s]');
ylabel('Izhod sistema');
grid on;


Priprava dataseta
y_train = y_train';
y_test = y_test';

% Pari vhod-izhod za trening
X_train = [u_train(2:end-1), u_train(1:end-2), -y_train(2:end-1), -y_train(1:end-2)];
Y_train = y_train(3:end);

X_test = [u_test(2:end-1), u_test(1:end-2), -y_test(2:end-1), -y_test(1:end-2)];
Y_test = y_test(3:end);

% Premešanje za boljšo generalizacijo treninga
perm = randperm(size(X_train, 1));
X_shifted = X_train(perm, :);
Y_shifted = Y_train(perm, :);


Nevronska mreža
% Nevronska mreža arhitektura - dva layerja sigmoide in linearna funckija
net = feedforwardnet([15, 10, 5]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'purelin';
net.trainParam.showWindow = false;  
net.trainParam.goal = 1e-7;  % Cilj, da mreža doseže takšn MAE

[net, ~] = train(net, X_shifted', Y_shifted'); % Trening mreže net - nadzorovano učenje
% Vrne treniran network net in log treninga (spustimo)

% Priprava inputov in outputov za sim
X_sim = X_test(1, :);
Y_sim = [];

% Simuliramo odziv mreže
for i = 3:length(u_test)   % Od tretjega sampla naprej
    y = net(X_sim');
    Y_sim = [Y_sim; y];
    X_sim = [u_test(i), u_test(i-1), -y, X_sim(3)];   % Update vhoda za naslednji korak
end

 % MAE
mae_nn = mean(abs(Y_test - Y_sim));
fprintf('MAE nevronske mreže: %.4f\n', mae_nn);

% Plot actual vs simulated neural network response
figure('Position', [100, 100, 1200, 800]);

plot(Y_test);
title("Primerjava odzivov - simuliran vs NN");
hold on;
plot(Y_sim);

xlabel("Sample");
ylabel("Odziv [V]");
legend('Dejanski odziv', 'Napovedan odziv');
grid on;


save('APRBS_signal.mat', 'X_shifted', "Y_shifted", "Y_test", "y_test", "u_test");


load("APRBS_signal.mat")

TS
% Replace your entire TS section with this simple code:

fprintf('=== SIMPLE TAKAGI-SUGENO MODEL ===\n');

%% Step 1: Use the correct data (from centered signals)
% Problem: You're using y_train from BEFORE centering u_train
% Solution: Re-simulate with centered signals or use the prepared data

% Use your existing prepared data
clustering_data = X_shifted(:, 3:4);  % Past outputs for clustering
regression_data = [X_shifted, ones(size(X_shifted, 1), 1)];  % All vars + constant

fprintf('Using %d training samples\n', size(X_shifted, 1));

%% Step 2: Check data diversity and adjust clusters
unique_points = unique(round(clustering_data, 2), 'rows');
max_clusters = size(unique_points, 1) - 1;
num_clusters = min(4, max_clusters);  % Start with 4 or less

fprintf('Unique data points: %d, Using %d clusters\n', size(unique_points, 1), num_clusters);

if num_clusters < 2
    error('Not enough data diversity. Check your APRBS signal generation.');
end

%% Step 3: FCM Clustering  
fprintf('Performing FCM clustering...\n');

fprintf('clustering_data size: [%d, %d]\n', size(clustering_data));
fprintf('num_clusters: %d\n', num_clusters);


[U, centers, ~] = fcm(clustering_data, num_clusters, [2, 100, 1e-5, 0]);

sigmas = zeros(num_clusters, 2);

for i = 1:num_clusters
    % Get membership values for cluster i (transpose to get column vector)
    memberships = U(i, :)';  % [N x 1]
    
    % Normalize memberships
    memberships = memberships / sum(memberships);
    
    % Calculate weighted mean for this cluster
    weighted_mean = sum(memberships .* clustering_data, 1);  % [1 x 2]
    
    % Calculate weighted variance for each dimension
    for dim = 1:2
        diff_sq = (clustering_data(:, dim) - weighted_mean(dim)).^2;
        weighted_variance = sum(memberships .* diff_sq);
        sigmas(i, dim) = sqrt(weighted_variance + 0.1);  % Add small value for stability
    end
end

% Use reasonable sigmas (not the huge ones from FCM)
spreads = sigmas;


%% Step 4: Calculate Membership Functions - FIXED
fprintf('Calculating membership functions...\n');
membership_functions = zeros(num_clusters, size(clustering_data, 1));

for i = 1:num_clusters
    diff = clustering_data - centers(i, :);  % Now should work: [4198×2] - [1×2]
    membership_functions(i, :) = exp(-0.5 * sum((diff ./ spreads(i, :)).^2, 2))';
end

% Normalize memberships
total_membership = sum(membership_functions, 1);
for i = 1:num_clusters
    membership_functions(i, :) = membership_functions(i, :) ./ total_membership;
end

fprintf('Membership functions calculated successfully.\n');

%% Step 5: Learn Rule Parameters (Weighted Least Squares)
fprintf('Learning rule parameters...\n');
rule_parameters = zeros(num_clusters, 5);  % 4 inputs + 1 constant

for i = 1:num_clusters
    weights = membership_functions(i, :)';
    W_matrix = diag(weights);
    
    % Weighted least squares: theta = (X'*W*X)^(-1) * X'*W*Y
    rule_parameters(i, :) = (regression_data' * W_matrix * regression_data) \ ...
                            (regression_data' * W_matrix * Y_shifted);
end

%% Step 6: Simulation
fprintf('Testing TS model...\n');
y_ts_sim = zeros(length(Y_test), 1);

for j = 1:length(Y_test)
    current_state = X_test(j, 3:4);  % Past outputs
    current_input = [X_test(j, :), 1];  % All inputs + constant
    
    % Calculate firing strengths
    firing_strengths = zeros(num_clusters, 1);
    for i = 1:num_clusters
        diff = current_state - centers(i, :);
        firing_strengths(i) = exp(-0.5 * sum((diff ./ spreads(i, :)).^2));
    end
    
    % Normalize firing strengths
    firing_strengths = firing_strengths / sum(firing_strengths);
    
    % Calculate rule outputs and final prediction
    rule_outputs = rule_parameters * current_input';
    y_ts_sim(j) = sum(firing_strengths .* rule_outputs);
end

%% Step 7: Evaluate Performance
mae_ts = mean(abs(Y_test - y_ts_sim));
rmse_ts = sqrt(mean((Y_test - y_ts_sim).^2));

fprintf('TS Model Performance:\n');
fprintf('  MAE: %.4f\n', mae_ts);
fprintf('  RMSE: %.4f\n', rmse_ts);

%% Step 8: Plot Results
figure('Position', [100, 100, 1200, 600]);
plot(Y_test, 'b-', 'LineWidth', 1.5);
hold on;
plot(y_ts_sim, 'r--', 'LineWidth', 1.5);
xlabel('Sample');
ylabel('Output');
title('Takagi-Sugeno Model Results');
legend('Actual', 'TS Model', 'Location', 'best');
grid on;

fprintf('=== TS MODEL COMPLETED ===\n\n');

% Store the TS model parameters for later use in your Convert function
C = centers;     % For your Convert function
O = spreads;     % For your Convert function  
W = rule_parameters(:, 1:4);  % Weights (without constant term)
b = rule_parameters(:, 5);    % Biases (constant terms)


Naloga 6
N = 4000;        % ŠT. aprbs samplov
%amplitude = 2;   % amplituda napetosti
Ts = 0.01;       % Sampling 
Th = 0.15;       % Hrizont spreminjanja aprbs signala
padding = 200;   % paddanje signala pred/po signalu
% Referenčni sistem (1. reda)
casovniKorak = 0.01;    % sampling

% Prenosna funkcija
s = tf('s'); % s je spremenljivka prenosne funckije
G_ref = 5 / (s + 5);  % Prenosna funkcija referenčnega sistema
sys_ref = c2d(ss(G_ref), Ts, 'zoh');  % Diskretizacija referenčnega sistema, samplanje s 0.01
%[A_ref, B_ref, C_ref, ~] = ssdata(sys_ref);


% Matrike prostora stanj - diskretiziran referenčni sistem sys_ref
A_ref = sys_ref.A;
B_ref = sys_ref.B;
C_ref = sys_ref.C;


% Ustvarjanje referenčnega signala
cas = 0:casovniKorak:99.99;
steviloTock = length(cas);   % Število korakov v signalu
referencniSignal = zeros(1, steviloTock);
amplitude = [11, 33, 55, 77, 99, 77, 55, 33, 11, 1];  % Zaporedje amplitud signala - mau random
casovniInterval = steviloTock / length(amplitude);   % Interval za vsako amplitudo

for i = 1:length(amplitude)
    % Začetni in končni indeks posamezne amplitude
    zacetek = (i - 1) * casovniInterval + 1;
    konec = i * casovniInterval;
    referencniSignal(zacetek:konec) = amplitude(i);
end
figure;
plot(cas, referencniSignal);
title("Referencni signal");
xlabel("Cas [s]");
ylabel("Kot [deg]");


% Simulation parametri
casovniKorak = 0.01;
% Parametri simulacije
Ts = 0.01;  % Vzorec (Sample time)
N_sim = 10000;  % Število simulacijskih korakov
H = 3;  % Število upoštevanih členov




% Inicializacija stanj in vhodnih signalov
x_ref = 0; % Začetno stanje reference
x_model = [0; 0]; % začetno stanje modela sistema
x_process = [0; 0]; % začetno stanje procesa sistema
u = zeros(1, N_sim); % Vhodni vektor na nuli
y_model = zeros(1, N_sim);          % model
y_process = zeros(1, N_sim);        % proces
y_ref = zeros(1, N_sim);            % referencni signal


% Integration and derivative component ADD
xi = 0;
Ki = 0.001;
Kd = 10;

previous_error = 0;

% Koeficienti za funkcijski regulator
for k = 3:N_sim
    % Izračun napake
    y_ref(k) = C_ref * x_ref;  % Izhod referenčnega sistema -> C matrika * referenčni vhod
    e = referencniSignal(k) - y_process(k-1);  % Napaka glede na proces, na zaćetku je y_process = 0


    xi = xi + e;

    % Calculate derivative term
    derivative_term = Kd * (e - previous_error);

    % Update previous error
    previous_error = e;


    % Izračunovanje matrik sistema na podlagi TS modela za trenutno stanje modela
    [A_m, B_m, C_m, R_m] = Convert(C, O, W, b, [-y_process(k-1); -y_process(k-2)]);

    % Prediktivni regulator -> regulirni zakon iz worda
    G0 = C_m * (A_m^H - eye(size(A_m))) * pinv(A_m - eye(size(A_m))) * B_m;
    G = pinv(G0) * (1 - A_ref^H);

   
    % Posodobitev stanj za referenčni, modelni in procesni sistem
    x_ref = A_ref * x_ref + B_ref * referencniSignal(k);
    x_model = A_m * x_model + B_m * u(k) + R_m;
    y_model(k) = C_m * x_model;

     % Krmilni signal - regulirni zakon
    u(k) = G * e + xi * Ki + derivative_term + pinv(G0) * y_model(k-1) - pinv(G0) * C_m * (A_m^H) * x_model - pinv(B_m) * R_m;


    % Procesni model (Helicrane)
    [x_process(2), x_process(1)] = helicrane(u(k), x_process);
    y_process(k) = x_process(2);
end

% Prikaz rezultatov
figure;
subplot(2, 1, 1);
plot(cas, u);
title('Vhodni signal');
xlabel('Čas [s]');
ylabel('Napetost [V]');

subplot(2, 1, 2);
plot(cas, referencniSignal, 'k--');
hold on;
plot(cas, y_ref, 'b');
plot(cas, y_process, 'g');
%legend('Referenčni signal', 'Modelni izhod', 'Izhod procesa');
title('Odziv sistema');
xlabel('Čas [s]');
ylabel('Kot [stopinje]');

Formule
% Funckija za poganjanje modela sistema - helikotper
function [y] = simulateHelicrane(x, u, t)
% x - stanje sistema, u - vhod v sistem, t - časovni vektor
    y = zeros(1, length(t)); % Nastavek izhodnega vektorja
    for i = 1:length(u)
        Fm = u(i);
        [fi_, fip_] = helicrane(Fm, x); % Klicanje funckije z trenutnim vhodom Fm in stanjem x, vrača pozicijo in hitrost
        x = [fip_, fi_];
        y(i) = fi_; % pozicijo shranit v izhodni vektor
    end
end

% Funkcija za generirat APRBS signal
function [u, t] = generateAPRBS(N, amplitude, Ts, Th, padding)
% N - število samplov v generiranem signalu, amplituda signala, Ts -
% sampling signala, Th - časovni horizont spremembe signala, padding -
% paddanje signala
    prbs = randi([0, 1], N, 1) * 2 - 1; % geneirira N APRBS signala od -1 do 1

    % Določit število korakov v korakov v signalu: časovni horizont Th /
    % časovna konstanta Ts
    steps = round(Th / Ts);
    
    % Nastavitev naključnih vrednosti amplitude
    u = zeros(N, 1);
    current_amplitude = amplitude * rand;
    

    % Iteriranje skozi korake
    for i = 1:steps:N
        segment_length = min(steps, N - i + 1);
        u(i:i + segment_length - 1) = current_amplitude;
        
        if rand > 0.5
            current_amplitude = amplitude * rand;
        end
    end

    % Paddanje signala z 0 na začetku
    u = [zeros(padding, 1); u];

    % Časovni vektor
    t = (0:length(u)-1) * Ts;
end

% Current code has incomplete gradient calculations
% Fix the TS_train function:

function [C, O, W, b] = TS_train(C, O, W, b, X_train_normalized, Y_train_normalized, learning_rate, epochs)
    for epoch = 1:epochs
        total_loss = 0;  
        for i = 1:size(X_train_normalized, 1)
            X = X_train_normalized(i, :);
            Y = Y_train_normalized(i);

            % Calculate membership values
            membership_values = exp(-0.5 * sum((X(3:4) - C).^2 ./ (O.^2), 2));
            
            % Prevent division by very small numbers
            if sum(membership_values) < 1e-6
                membership_values = membership_values + 1e-6;
            end
            
            % Normalize membership values
            membership_values = membership_values / sum(membership_values);

            % Calculate rule outputs
            rule_outputs = W * X' + b; % R x 1 vector
            
            % Calculate predicted output
            y_predicted = sum(membership_values .* rule_outputs);

            % Calculate error
            error = Y - y_predicted;
            loss = error^2;
            total_loss = total_loss + loss;

            % Calculate gradients (FIXED)
            % For centers
            diff_c = (X(3:4) - C) ./ (O.^2); % R x 2
            grad_C = zeros(size(C));
            for r = 1:size(C,1)
                grad_C(r,:) = diff_c(r,:) * membership_values(r) * ...
                    (rule_outputs(r) - y_predicted);
            end
            
            % For spreads
            diff_o = ((X(3:4) - C).^2) ./ (O.^3); % R x 2
            grad_O = zeros(size(O));
            for r = 1:size(O,1)
                grad_O(r,:) = diff_o(r,:) * membership_values(r) * ...
                    (rule_outputs(r) - y_predicted);
            end
            
            % For weights and biases
            grad_W = membership_values * X; % R x 4
            grad_b = membership_values; % R x 1

            % Update parameters
            C = C - learning_rate * (-2 * error * grad_C);
            O = O - learning_rate * (-2 * error * grad_O);
            W = W - learning_rate * (-2 * error * grad_W);
            b = b - learning_rate * (-2 * error * grad_b);
            
            % Ensure O stays positive
            O = max(O, 0.1);
        end
        
        total_loss = total_loss / size(X_train_normalized, 1);
        if mod(epoch, 10) == 1
            fprintf('Epoch %d, Loss: %.6f\n', epoch, total_loss);
        end
    end
end

% Funkcija za napoved izhoda TS modela
% Fix the TS_eval function:

function [Y] = TS_eval(C, O, W, b, X)
    % Handle both single sample and multiple samples
    if size(X, 1) == 1  % Single sample as row vector
        X = X';  % Convert to column vector
    end
    
    if size(X, 2) == 1  % Single sample as column vector
        num_samples = 1;
        X_samples = X;
    else  % Multiple samples
        num_samples = size(X, 2);
        X_samples = X;
    end
    
    Y = zeros(1, num_samples);
    
    for i = 1:num_samples
        if num_samples == 1
            x_current = X_samples;
        else
            x_current = X_samples(:, i);
        end
        
        % Calculate membership values (only for positions 3:4)
        membership_values = exp(-0.5 * sum((x_current(3:4)' - C).^2 ./ (O.^2), 2)); 
        
        % Prevent division by zero
        if sum(membership_values) < 1e-6
            membership_values = membership_values + 1e-6;
        end
        
        % Normalize membership values
        membership_values = membership_values / sum(membership_values);
        
        % Calculate rule outputs (use all 4 inputs)
        rule_outputs = W * x_current + b; 
        Y(i) = sum(membership_values .* rule_outputs); 
    end
end


function [A_state, B_state, C_state, R_state] = Convert(centers, spreads, weights, offsets, input)

    % Prevajanje TS modela v prostor stanj

    numRules = size(centers, 1);  % Število funkcij
    [A_state, B_state, C_state, R_state] = deal(zeros(2, 2), zeros(2, 1), zeros(1, 2), zeros(2, 1));

    membershipValues = exp(-0.5 * sum((input' - centers).^2 ./ (spreads.^2), 2));  % Gaussian membership
    membershipValues = membershipValues / sum(membershipValues);  % Normalizcija membershipov

    % Sestavitev matrik stanja
    for ruleIndex = 1:numRules
        A_rule = [0, -weights(ruleIndex, 4); 1, -weights(ruleIndex, 3)];  % 2x2, uteži po diagonali -> Dinamika sistema
        B_rule = [weights(ruleIndex, 2); weights(ruleIndex, 1)]; % 2x1 -> vpliv vhoda na stanje
        C_rule = [0, 1];
        R_rule = [0; offsets(ruleIndex)];  % Deviacija dinamike za funkcijo

        % Posodobitev matrik prostora stanj za posamezno pravilo
        A_state = A_state + membershipValues(ruleIndex) * A_rule;
        B_state = B_state + membershipValues(ruleIndex) * B_rule;
        C_state = C_state + membershipValues(ruleIndex) * C_rule;
        R_state = R_state + membershipValues(ruleIndex) * R_rule;
    end
    input=0;
    
end


