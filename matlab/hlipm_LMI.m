clc; clear; close all;
set(groot, 'DefaultLegendInterpreter', 'latex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultAxesFontSize', 40);
set(groot, 'DefaultLineLineWidth', 2);

plot_colors = { [0 0.4470 0.7410] ...
                [0.8500 0.3250 0.0980] ...
                [0.9290 0.6940 0.1250] ...
                [0.4940 0.1840 0.5560] ...
                [0.4660 0.6740 0.1880] ...
                [0.3010 0.7450 0.9330] ...
                [0.6350 0.0780 0.1840] };

%% Reference parameters
r_bar = 0.15;
T_ref = 1.2;
z_c = 0.58;
omega = sqrt(9.81/z_c);
u_bar = 0.075;
v_bar = omega * r_bar * (cosh(omega*T_ref)+1) / sinh(omega*T_ref);
xi = r_bar*omega/(v_bar/omega-r_bar);

%% Plant
A = [0, 1; omega^2, 0];
B = [0; -omega^2];

%% Controller selection (linear feedback)
alpha = 1.02*omega;
yalmip('clear');
opts = sdpsettings('solver', 'mosek', 'verbose', 0);
fz = 1e-6;

q_11 = sdpvar(1,1);
q_12 = sdpvar(1,1);
q_22 = sdpvar(1,1);
Q = [q_11, q_12; q_12, q_22];

Delta = [ (exp(2*(omega-alpha)*T_ref)-1) * q_22 + ...
            4 * exp(-2*alpha*T_ref) * xi * (xi*q_11 - q_12*exp(omega*T_ref)), ...
          2*exp(-2*alpha*T_ref)*xi*q_11 + ...
            (exp(-(omega+2*alpha)*T_ref)-1)*q_12 ; ...
          2*exp(-2*alpha*T_ref)*xi*q_11 + ...
            (exp(-(omega+2*alpha)*T_ref)-1)*q_12 , ...
          (exp(-2*alpha*T_ref)-1) * q_11];

%% Initialization
%%%%%%%%%%%%%%
% SATURATION %
%%%%%%%%%%%%%%
W = sdpvar(1,2);
Y = sdpvar(1,2);
U = sdpvar(1,1);
X = sdpvar(1,1);
constr = [ Q >= fz*eye(2); norm(Q) <= 10; ...
           norm(W) <= 10; norm(Y) <= 10; norm(U) <= 10; norm(X) <= 10; ...
           U >= fz; ...
           Delta <= -fz*eye(2); ...
           [ 2*alpha*Q+(A*Q+B*W)+(A*Q+B*W)' , (W+Y)'+B*(X-U) ; ...
                     W+Y+(X-U)*B'       ,        2*(X-U)     ] <= -fz*eye(3); ...
           [ u_bar^2, Y; Y', Q ] >= 0 ];
       
optimize(constr, -log(det(Q)), opts);
[primalfeas, dualfeas] = checkset(constr);

Q = double(Q);
K = double(W)/Q;
L = double(X)/double(U);

feas = (all(primalfeas >= -fz) && min(eig(Q)) > 0);

if feas
    disp("Feasible LMI")
else
    disp("Infeasible LMI");
end

%% Simulation
TSPAN = [0 5];
JSPAN = [0 100];
rule = 1;
opts = odeset('RelTol',1e-6,'MaxStep',1e-3);

x0 = [-1e-1; 0.12; (0.3)*T_ref];

[t j x] = HyEQsolver( @(x,t) f(x,A,B,K,r_bar,v_bar,L,u_bar), ...
                      @(x,t) g(x,r_bar,T_ref), ...
                      @(x,t) C(x,r_bar), ...
                      @(x,t) D(x,r_bar), ...
                      x0,TSPAN,JSPAN,rule,opts,'ode45');

ref = zeros(length(t),2);
for i=1:length(t)
    ref(i,:) = (expm(A*x(i,3))*[-r_bar; v_bar])';
end

e = x(:,1:2) - ref;
for i = 1:length(t)
    u(i) = sat(K*e(i,:)', u_bar);
end

V = zeros(length(t),1);
for i = 1:length(t)
    V(i) = exp(alpha*x(i,3))*e(i,:)/Q*e(i,:)';
end
%% Plotting
fig1 = figure(1);
fig1.Color = 'w';

subplot(211);
plotarc(t,j,[x(:,1), ref(:,1)],[],[],{'-','linewidth',2},{':','linewidth',2});
legend('Robot', 'Modified reference');
title('Position');

subplot(212);
plotarc(t,j,[x(:,2), ref(:,2)],[],[],{'-','linewidth',2},{':','linewidth',2});
legend('Robot', 'Modified reference');
title('Velocity');


fig2 = figure(2);
fig2.Color = 'w';
plotarc(t,j,u,[],[],{'-','linewidth',2},{':','linewidth',2});
title('Input');

fig3 = figure(3);
fig3.Color = 'w';
plotarc(t,j,V,[],[],{'-','linewidth',2},{':','linewidth',2});
title('Lyapunov function');

fig4 = figure(4);
fig4.Color = 'w';
plotellisa(inv(Q), [0;0], '-', plot_colors{1});
hold on;
plot(e(:,1), e(:,2), 'Color', plot_colors{2});
title('Region of attraction');

%% Save K
save(strcat(getenv('HOME'), '/devel/src/hybrid-biped/data/data.mat'), ...
    "K", "x0");

%% External functions

function xdot = f(x,A,B,K,r_bar,v_bar,L,u_bar)
    
    x_p = x(1:2);
    tau = x(3);
    ref = expm(A*tau)*[-r_bar; v_bar];
    e = x_p - ref;
    
    %%%%%%%%%%%%%%%%%
    % NO SATURATION %
    %%%%%%%%%%%%%%%%%
%     xdot = [A*x_p + B*K*eye(2)*e; 1];
    
    %%%%%%%%%%%%%%
    % SATURATION %
    %%%%%%%%%%%%%%
    xdot = [A*x_p + B*sat(K*e, u_bar); 1];

end

function xplus = g(x,r_bar,T)
    
    tau = x(3);
    k = floor(tau/T);
    tau_plus = [tau-k*T, tau-(k+1)*T];
    [~, idx] = min(abs(tau_plus));
    xplus = [x(1)-2*r_bar; x(2); tau_plus(idx)];

end

function vC = C(x,r_bar)
    vC = (x(1) <= r_bar);
end

function vD = D(x,r_bar)
    vD = (x(1) >= r_bar);
end
