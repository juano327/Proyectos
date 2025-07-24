clear;clc;close all
%matriz dh parametros
% Define the initial joint positions
q = [0,0,0,0,0];
dh = [
    % theta   d      a      alpha sigma
    0.000  -0.160  0.160  pi/2   0.000;
    -pi/2   -0.080  0.000 -pi/2   0.000;
    pi/2    0.000  0.440  0.000  0.000;
    0.000   0.000  0.480  0.000  0.000;
    -pi/2    0.000  0.270  0.000  0.000
    ];

% Crear el objeto del robot usando la matriz DH
R = SerialLink(dh, 'name', 'Atalante');

% Desplazamiento articular
R.offset = [0 -pi/2 pi/2 0 -pi/2]; % Offset inicial

% Definir la herramienta (pie) a una distancia bajo el tobillo (0.1 metro)
R.tool = transl(0, 0, 0);
R.base = transl(0, 0, 0) * trotx(-pi/2);

% Límites articulares actualizados
R.qlim(1,1:2) = [-30,  45] * pi/180; % Aducción 0-30°, Abducción 0-45°
R.qlim(2,1:2) = [-45,  5]  * pi/180; % Rotación externa de cadera 0-45° y rotación interna
R.qlim(3,1:2) = [-30, 120] * pi/180; % Flexión muslo-torso 0-120°, extensión 0-30°
R.qlim(4,1:2) = [-5, 135]  * pi/180; % Rodilla flexión 0-135°, extensión 0-5°
R.qlim(5,1:2) = [-50,  20] * pi/180; % Dorsiflexión 0-20°, flexión plantar 0-50°

% Robot visualization
R.plot(q, 'scale', 0.4, 'workspace', [-1.2 1.2 -1.2 1.2 -1.2 1.2]);

% Display the joint manipulation interface
R.teach();

T = eye(4);
     for i = 1:size(dh,1)
        A = trotz(dh(i))*transl(dh(i,3),0,dh(i,2))*trotx(dh(i,4));
       T = T*A;
     end

