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


APRBS

N = 4000;        % ŠT. aprbs samplov
amplitude = 2; % 2   % amplituda napetosti
Ts = 0.01;       % Sampling 
Th = 0.15;       % Hrizont spreminjanja aprbs signala
padding = 200;   % paddanje signala pred/po signalu

% generiranje aprbs signala
[u_train, t_train] = generateAPRBS(N, amplitude, Ts, Th, padding);
[u_test, t_test] = generateAPRBS(N, amplitude, Ts, Th, padding);

% simulacija odziva na aprbs
y_train = simulateHelicrane([0 0],u_train,t_train);
y_test = simulateHelicrane([0 0],u_test,t_test);


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
% Parametri TS modela
R = 10; % Število podfunkcij
C = rand(R, 2);  % Centri funkcij
O = rand(R, 1);  % Širine podfunkcij
W = rand(R, 4);  % Uteži linearnih funkcij
b = rand(R, 1);  % Biasi linearnih funkcij
learning_rate = 0.0001; % Learning rate
epochs = 100; % Število učnih epoh

% Membership funckija pove, koliko posamezno pravilo doprinese k končnemu
% izhodu funkcije

% trening TS modela
[C, O, W, b] = TS_train(C, O, W, b, X_shifted, Y_shifted, learning_rate, epochs);


% SIMULATION MODE
Y_simulated = zeros(size(Y_test));



for i = 1:length(Y_test)
    if i == 1
        y_prev1 = y_test(i+1); % y_test(2)
        y_prev2 = y_test(i);   % y_test(1)
    elseif i == 2
        y_prev1 = Y_simulated(i-1); % Y_simulated(1) which predicts y_test(3)
        y_prev2 = y_test(i);        % y_test(2)
    else
        y_prev1 = Y_simulated(i-1);
        y_prev2 = Y_simulated(i-2);
    end

    % Construct the input vector
    x_current = [u_test(i+1), u_test(i), -y_prev1, -y_prev2];

    % Predict the current output
    y_predicted_current = TS_eval(C, O, W, b, x_current');

    % Store the predicted output
    Y_simulated(i) = y_predicted_current;
end

% Calculate MAE for simulation
mae_ts_sim = mean(abs(Y_test - Y_simulated'));
fprintf('Takagi-Sugeno Model MAE (Simulation): %.4f\n', mae_ts_sim);

% Plot simulation results
figure('Position', [100, 100, 1200, 800]);
plot(Y_test);
hold on;
plot(Y_simulated);
title('Primerjava simuliranega in dejanskega izhoda (TS)');
xlabel('Sample');
ylabel('Vrednost izhoda [V]');
legend('Dejanski izhod', 'Simulirani izhod modela');
grid on;
hold off;


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

for k = 3:N_sim
    % Izračun napake
    y_ref(k) = C_ref * x_ref;  % Izhod referenčnega sistema -> C matrika * referenčni vhod
    
    % Izračunovanje matrik sistema na podlagi TS modela za trenutno stanje
    [A_m, B_m, C_m, R_m] = Convert(C, O, W, b, [-y_process(k-1); -y_process(k-2)]);
    
    % MPC Control Law (replace PID)
    % Calculate step response matrix G0 for horizon H
    G0 = C_m * (A_m^H - eye(size(A_m))) * pinv(A_m - eye(size(A_m))) * B_m;
    
    % Calculate control gain G
    G = pinv(G0) * (1 - A_ref^H);
    
    % MPC control signal calculation
    error = referencniSignal(k) - y_process(k-1);  % Current tracking error
    
    % MPC control law (similar to main6.m)
    u(k) = G * error + pinv(G0) * y_model(k-1) - pinv(G0) * C_m * (A_m^H) * x_model - pinv(B_m) * R_m;
    
    % Update states
    x_ref = A_ref * x_ref + B_ref * referencniSignal(k);
    x_model = A_m * x_model + B_m * u(k) + R_m;
    y_model(k) = C_m * x_model;
    
    % Update process (Helicrane)
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


% Funkcija treninga TS modela
function [C, O, W, b] = TS_train(C, O, W, b, X_train_normalized, Y_train_normalized, learning_rate, epochs)
    for epoch = 1:epochs % Iteracije skozi trening epohe
        total_loss = 0;  
        for i = 1:size(X_train_normalized, 1) % Iteracija skozi trening sample
            X = X_train_normalized(i, :);
            Y = Y_train_normalized(i);

            % Izračun pripadnosti posameznim funckijam
            % Računa glede na razdaljo med vhodnimi značilkami (3, 4) in
            % centrom, normalizirano z širino O
            membership_values = exp(-0.5 * sum((X(3:4) - C).^2 ./ (O.^2), 2)); %-> Gaussova funkcija pripadnosti

            % Prevent division by very small numbers
            % fallback - da se ne deli z zelo majhnimi vrednostmi
            if sum(membership_values) < 1e-3
                membership_values = membership_values + 1e-3;
            end

            % Normalizacija pripadnosti
            membership_values = membership_values / sum(membership_values);

            % Izračun napovedane funkcije
            % y = ax + b
            % Napoved funkcij kot utežena vsota podfunkcij
            rule_outputs = W * X' + b; % R x 1 vector

            % izhodi pravil - uteženo z membership vrednostmi
            y_predicted = sum(membership_values .* rule_outputs);

            % Izračun napake med napovedano in ground truth funkcijo
            error = Y - y_predicted;
            loss = error^2;

            total_loss = total_loss + loss;

            % Odvodi glede na vse parametre
        
            grad_C = (X(3:4) - C) ./ O.^2 .* membership_values .* (rule_outputs - sum(membership_values .* rule_outputs)) / sum(membership_values);
            grad_O = ((X(3:4) - C).^2 ./ O.^3) .* membership_values .* (rule_outputs - sum(membership_values .* rule_outputs)) / sum(membership_values);
            grad_W = membership_values * X;
            grad_b = membership_values;

            % Posodobitev vrednosti matrik/vektorjev s korakom lr
            C = C - learning_rate * (-2 * error * grad_C);
            O = O - learning_rate * (-2 * error * grad_O);
            W = W - learning_rate * (-2 * error * grad_W);
            b = b - learning_rate * (-2 * error * grad_b);
        end
        total_loss = total_loss / size(X_train_normalized, 1);
        fprintf('Epoch %d, Loss: %.4f\n', epoch, total_loss);
    end
end

% Funkcija za napoved izhoda TS modela
function [Y] = TS_eval(C, O, W, b, X)
    num_samples = size(X, 2);
    Y = zeros(1, num_samples);
    
    % Loop skozi vse sample v signalu za primerjanje
    for i = 1:num_samples

        % Izračun pripadnosti podfunkcijam - gauss
        membership_values = exp(-0.5 * sum((X(3:4, i)' - C).^2 ./ (O.^2), 2)); 
        
        % Normalizacija pripadnosti podfunkcijam
        membership_values = membership_values / sum(membership_values);
        

        % Izračun posameznih funkc
        rule_outputs = W * X(:, i) + b; 
        Y(i) = sum(membership_values .* rule_outputs); 
    end
end


function [A_state, B_state, C_state, R_state] = Convert(centers, spreads, weights, offsets, input)
    % Prevajanje TS modela v prostor stanj
    
    numRules = size(centers, 1);  % Število funkcij
    
    % Initialize matrices
    A_state = zeros(2, 2);
    B_state = zeros(2, 1);
    C_state = zeros(1, 2);
    R_state = zeros(2, 1);

    % Calculate membership values
    membershipValues = exp(-0.5 * sum((input' - centers).^2 ./ (spreads.^2), 2));  % Gaussian membership
    membershipValues = membershipValues / sum(membershipValues);  % Normalization

    % Weighted combination of rule matrices
    for ruleIndex = 1:numRules
        A_rule = [0, -weights(ruleIndex, 4); 1, -weights(ruleIndex, 3)];  % 2x2 system dynamics
        B_rule = [weights(ruleIndex, 2); weights(ruleIndex, 1)]; % 2x1 input influence
        C_rule = [0, 1]; % 1x2 output matrix
        R_rule = [0; offsets(ruleIndex)];  % 2x1 offset vector

        % Weighted sum of matrices
        A_state = A_state + membershipValues(ruleIndex) * A_rule;
        B_state = B_state + membershipValues(ruleIndex) * B_rule;
        C_state = C_state + membershipValues(ruleIndex) * C_rule;
        R_state = R_state + membershipValues(ruleIndex) * R_rule;
    end
end


