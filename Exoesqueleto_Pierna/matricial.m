clear
clc
close all


syms q1 q2 q3 q4 q5 d1 a1 d2 a3 a4 a5 real
n = sym('n',[3,1],'real');
o = sym('o',[3,1],'real');
a = sym('A',[3,1],'real');
p = sym('p',[3,1],'real');

TD = [n o a p];
TD(4,:) = [0 0 0 1];

% Matriz DH parámetros
dh = [
    % theta   d      a      alpha sigma
    0  d1  a1   pi/2  0.000;
    -pi/2  d2   0  -pi/2  0.000;
    pi/2   0  a3      0  0.000;
    0   0  a4      0  0.000;
    -pi/2   0  a5      0  0.000
    ];

% Crear el objeto del robot usando la matriz DH
R = SerialLink(dh, 'name', 'Atalante');

% Desplazamiento articular
% R.offset = [0 -pi/2 pi/2 0 -pi/2]; % Offset inicial

% Definir la herramienta (pie) a una distancia bajo el tobillo (0.1 metro)
% R.tool = transl(0, 0, 0);
% R.base = transl(0, 0, 0) * trotx(-pi/2);

q = [q1,q2,q3,q4,q5];

T06 = R.fkine(q).double;

fprintf('> Las 12 posibles ecuaciones de la CD (T1*T2*T3*T4*T5 = T):\n\n')

fprintf('>  (1): TD(1,1) = %s\n', T06(1,1))
fprintf('>  (2): TD(2,1) = %s\n', T06(2,1))
fprintf('>  (3): TD(3,1) = %s\n', T06(3,1))
fprintf('>  (4): TD(1,2) = %s\n', T06(1,2))
fprintf('>  (5): TD(2,2) = %s\n', T06(2,2))
fprintf('>  (6): TD(3,2) = %s\n', T06(3,2))
fprintf('>  (7): TD(1,3) = %s\n', T06(1,3))
fprintf('>  (8): TD(2,3) = %s\n', T06(2,3))
fprintf('>  (9): TD(3,3) = %s\n', T06(3,3))
fprintf('> (10): TD(1,4) = %s\n', T06(1,4))
fprintf('> (11): TD(2,4) = %s\n', T06(2,4))
fprintf('> (12): TD(3,4) = %s\n', T06(3,4))

fprintf('\n')

fprintf('> Dividiendo la ecuación (8) sobre la (7) se tiene que:\n')
%creo que lo de abajo está mal
%fprintf('> TD(2,3)/TD(2,1) = tan(q1)   =>   q1 = atan2(TD(2,3), TD(2,1))\n')
% y sería así:
fprintf('> TD(2,3)/TD(1,3) = tan(q1)   =>   q1 = atan2(TD(2,3), TD(1,3))\n')
fprintf('> IMPORTANTE: esto sirve solo si q2 != n*pi\n')
fprintf('\n')

fprintf('> Con la ecuación (9) se calcula q2, que por ser un acos() tendrá 2 posibles valores, el positivo y el negativo:\n')
fprintf('> q2 = acos(TD(3,3))\n')
fprintf('> Al ser un acos() debo evaluar si los 2 valores (positivo y negativo) son solución.\n')
fprintf('\n')

