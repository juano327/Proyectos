clear;
clc;

fprintf('robot Atalante\n');

% Matriz DH parámetros
dh = [
    0.000   -0.160  0.160  pi/2   0.000;
    -pi/2   -0.080  0.000 -pi/2   0.000;
    pi/2    0.000  0.440  0.000  0.000;
    0.000   0.000  0.480  0.000  0.000;
    -pi/2   0.000  0.270  0.000  0.000
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



%% Configuración 1: Totalmente extendido
q_extendido = [0, 0, 0, 0, 0];  % Todas las articulaciones alineadas

figure(1);  
R.plot(q_extendido, 'scale', 0.4, 'workspace', [-1.5 1.5 -1.5 1.5 -1.5 1.5]);
title('Configuración totalmente extendido');
drawnow;  

J1 = R.jacob0(q_extendido); %Jacobiano por la config 1
J1_reducido=J1(1:5,:)
det_J1 = det(J1_reducido); %determinante del J1
fprintf('Determinante para la configuración 1 (robot extendido): %.4f\n', det_J1);

%% Configuración 2: Totalmente plegado
q_plegado = [0, pi/2, pi/2, pi/2, pi/2];  % El extremo casi toca la base

figure(2); 
R.plot(q_plegado, 'scale', 0.4, 'workspace', [-1.5 1.5 -1.5 1.5 -1.5 1.5]);
title('Configuración totalmente plegado');
drawnow; 

J2 = R.jacob0(q_plegado); %Jacobiano por la config 2
J2_reducido=J2(1:5,:)
det_J2 = det(J2_reducido); %determinante del J2
fprintf('Determinante para la configuración 2 (robot plegado): %.4f\n', det_J2);



%% Configuración 3: Dos ejes de rotación alineados (ejemplo a nivel de la cadera y el codo)
q_alineado = [0, 0, pi/2, 0, 0];  % Alineación en las articulaciones

figure(3);  
R.plot(q_alineado, 'scale', 0.4, 'workspace', [-1.5 1.5 -1.5 1.5 -1.5 1.5]);
title('Configuración con dos ejes de rotación alineados');
drawnow;  

J3 = R.jacob0(q_alineado); %Jacobiano por la config 3
J3_reducido=J3(1:5,:)
det_J3 = det(J3_reducido); %determinante del J3
fprintf('Determinante para la configuración 3 (robot alineado): %.4f\n', det_J3);
