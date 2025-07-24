clear 
clc
close all

% Matriz DH
dh = [
    % theta   d      a      alpha sigma
    0.000  -0.160  0.160  pi/2   0.000;
    -pi/2   -0.080  0.000 -pi/2   0.000;
    pi/2    0.000  0.440  0.000  0.000;
    0.000   0.000  0.480  0.000  0.000;
    -pi/2    0.000  0.270  0.000  0.000
    ];

R = SerialLink(dh,'name','Atalante');

% Herramienta y base
R.tool = transl(0 , 0, 0); 
R.base = transl(0,0,0) * trotx(pi/2);

% Límites articulares
R.qlim(1,1:2) = [-30,  45]*pi/180; % Aducción/Abducción
R.qlim(2,1:2) = [-45,  5]*pi/180;  % Rotación externa/interna
R.qlim(3,1:2) = [-30,  120]*pi/180; % Flexión/extensión de la cadera
R.qlim(4,1:2) = [-135, 5]*pi/180;  % Rodilla flexión/extensión
R.qlim(5,1:2) = [-50,  20]*pi/180; % Dorsiflexión/Flexión plantar

% Muestreo de valores articulares dentro de los límites
n = 10; % Número de muestras por articulación
q1 = linspace(R.qlim(1,1), R.qlim(1,2), n);
q2 = linspace(R.qlim(2,1), R.qlim(2,2), n);
q3 = linspace(R.qlim(3,1), R.qlim(3,2), n);
q4 = linspace(R.qlim(4,1), R.qlim(4,2), n);
q5 = linspace(R.qlim(5,1), R.qlim(5,2), n);

% Crear una matriz para almacenar todas las posiciones posibles
workspace = [];

% Bucle sobre todas las combinaciones posibles de las configuraciones
for i1 = 1:n
    for i2 = 1:n
        for i3 = 1:n
            for i4 = 1:n
                for i5 = 1:n
                    % Vector articular actual
                    q = [q1(i1), q2(i2), q3(i3), q4(i4), q5(i5)];
                    
                    % Calcular la posición del extremo usando cinemática directa
                    T = R.fkine(q);
                    
                    % Extraer la posición del extremo
                    pos = T.t';
                    
                    % Almacenar la posición
                    workspace = [workspace; pos];
                end
            end
        end
    end
end

% Graficar el espacio de trabajo
figure;
plot3(workspace(:,1), workspace(:,2), workspace(:,3), 'b.');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Espacio de trabajo del robot');
grid on;
axis equal;
