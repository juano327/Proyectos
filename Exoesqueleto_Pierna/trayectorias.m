clc;
clear;
close all;

% Configuración DH del robot
dh = [
    % theta   d      a      alpha sigma
    0.000    -0.160  0.160  pi/2   0.000; 
   -pi/2    -0.080  0.000 -pi/2   0.000; 
    pi/2    0.000  0.440  0.000  0.000; 
    0.000   0.000  0.480  0.000  0.000; 
   -pi/2    0.000  0.270  0.000  0.000
];

R = SerialLink(dh, 'name', 'Atalante');

% Desplazamiento articular inicial
R.offset = [0 -pi/2 pi/2 0 -pi/2]; 

% Definir la herramienta (pie) y la base
R.tool = transl(0, 0, 0);
R.base = transl(0, 0, 0) * trotx(-pi/2);

% Límites articulares 
R.qlim(1,1:2) = [-30,  45] * pi/180; % Aducción 0-30°, Abducción 0-45°
R.qlim(2,1:2) = [-45,  5]  * pi/180; % Rotación externa de cadera 0-45° y rotación interna
R.qlim(3,1:2) = [-120, 30] * pi/180; % Flexión muslo-torso 0-120°, extensión 0-30°
R.qlim(4,1:2) = [-5, 135]  * pi/180; % Rodilla flexión 0-135°, extensión 0-5°
R.qlim(5,1:2) = [-50,  20] * pi/180; % Dorsiflexión 0-20°, flexión plantar 0-50°

% Configuraciones iniciales q(n) para la trayectoria
q1 = [0, 0, -90, 90, 0] * pi/180;   % Posición inicial: sentado
q2 = [0, 0, -45, 45, 0] * pi/180;   % Primera posición: parando
q3 = [0, 0, 0, 0, 0] * pi/180;      % Segunda posición: parado
q4 = [0, 0, -15, 30, -10] * pi/180;   % Tercera posición: comienza a caminar
q5 = [0, 0, -15, 5, 10] * pi/180;   % Cuarta posición: caminando
q6 = [0, 0, 0, 0, 0] * pi/180;      % Posición final: parado

q_trajectory = [q1; q2; q3; q4; q5; q6; q6; q4; q5; q6; q6; q4; q5; q6; q6];

%%  Paso 1: Calcular las matrices de transformación T(n) en espacio cartesiano usando cinemática directa

T_1 = R.fkine(q1).double;
T_2 = R.fkine(q2).double;
T_3 = R.fkine(q3).double;
T_4 = R.fkine(q4).double;
T_5 = R.fkine(q5).double;
T_6 = R.fkine(q6).double;

%% Pregunta 3.a : Transformar al espacio articular los puntos del espacio cartesiano 
q1_articular = cin_inv(R, T_1, q1, true); 
q2_articular = cin_inv(R, T_2, q1_articular, true);
q3_articular = cin_inv(R, T_3, q2_articular, true);
q4_articular = cin_inv(R, T_4, q3_articular, true);
q5_articular = cin_inv(R, T_5, q4_articular, true);
q6_articular = cin_inv(R, T_6, q5_articular, true);

q_articular=[q1_articular; q2_articular; q3_articular; q4_articular; q5_articular;q6_articular];

%% Pregunta3.a: Interpolación q_articular

n_points = 20;  % Número de puntos por tramo de interpolación
trayectoria_articular = [
    jtraj(q1_articular, q2_articular, n_points);
    jtraj(q2_articular, q3_articular, n_points);
    jtraj(q3_articular, q4_articular, n_points);
    jtraj(q4_articular, q5_articular, n_points);
    jtraj(q5_articular, q6_articular, n_points);
    jtraj(q6_articular, q6_articular, n_points);
    jtraj(q6_articular, q4_articular, n_points);
    jtraj(q4_articular, q5_articular, n_points);
    jtraj(q5_articular, q6_articular, n_points);
    jtraj(q6_articular, q6_articular, n_points);
    jtraj(q6_articular, q4_articular, n_points);
    jtraj(q4_articular, q5_articular, n_points);
    jtraj(q5_articular, q6_articular, n_points);
    jtraj(q6_articular, q6_articular, n_points);];


%% Trayectoria por la pregunta 3.a 
figure (1)
title('Trayectoria en el espacio articular despues de la interpolacion en el espacio articular ')
R.plot(q1_articular, 'scale', 0.4); % Mostrar la posición inicial
R.plot(trayectoria_articular, 'scale', 0.4,'trail', {'r', 'LineWidth', 2});

%% PREGUNTA 3.b 

% Paso 1: Calcular las matrices de transformación T(n) en espacio cartesiano usando cinemática directa
T = zeros(4, 4, size(q_trajectory, 1));  % Inicializar T(n)
vector = zeros(6,size(q_trajectory, 1));
for i = 1:size(q_trajectory, 1)
    T = R.fkine(q_trajectory(i, :));
    vector(:,i)=[T.transl';T.tr2rpy'];
end

%% Paso 2: Interpolación en el espacio cartesiano con mstraj para obtener T(m)
NUMBER_OF_INITIAL_Q = size(q_trajectory, 1);
WP = vector(:,2:end)'; % Puntos de la trayectoria
QDMAX = []; % Velocidad mÃ¡xima de cada articulaciÃ³n (no usado)
TSEG=ones(1,NUMBER_OF_INITIAL_Q-1) * 2; % Tiempo de cada segmento
q0 = vector(:, 1)'; % PosiciÃ³n inicial
dt = 0.001;  % Intervalo de tiempo entre cada punto interpolado
TACC = 0.5; % Tiempo de aceleraciÃ³n
%T_mstraj_positions = mstraj(posiciones_cartesianas(2:end, :), [], [2, 2, 2, 2], posiciones_cartesianas(1, :), dt, 0);
%W=mstraj(vector(:,2:end)',[], [1, 1, 1, 1 ,1],vector(:,1)',dt,0);
W=mstraj(WP, QDMAX, TSEG, q0, dt, TACC);
% VerificaciÃ³n
disp('Verficación de punto inicial')
disp(vector(1,:))
disp(W(1,:))

% Adaptación de W
disp('tamaño de W antes de la adaptación:')
disp(size(W))
MAX_W_SIZE = 100;
if size(W, 1) > MAX_W_SIZE
    w_step = floor(size(W, 1) / MAX_W_SIZE);
    W = W(1:w_step:end, :);
end
disp('tamaño de W despues de la adaptación:')
disp(size(W))

W=W';
NUMBER_OF_POINTS = size(W, 2);
% Inicializar T_m para almacenar matrices de transformación homogéneas (4x4xN)
T_m = zeros(4,4,NUMBER_OF_POINTS);

for i = 1:size(W, 2)
    % Extraer traslaciones y rotaciones de cada columna de W
    x = W(1, i);  % traslación en x
    y = W(2, i);  % traslación en y
    z = W(3, i);  % traslación en z
    roll = W(4, i);  % rotación en roll
    pitch = W(5, i); % rotación en pitch
    yaw = W(6, i);   % rotación en yaw
    
    % Crear la matriz de rotación usando rpy2tr
    Rot = rpy2tr(roll, pitch, yaw);
    
    % Formar la matriz de transformación T(m) con la rotación y traslación
    T_m(:, :, i) = transl(x, y, z) * Rot;

end

% Initialiser la matrice positions pour stocker les translations
positions = zeros(size(T_m, 3), 3);  % Taille : nombre de matrices T_m x 3

% Extraire la translation de chaque matrice T_m
for i = 1:size(T_m, 3)
    positions(i, :) = T_m(1:3, 4, i)';  % Transpose pour stocker en tant que ligne
end

%% Transformar al espacio articular

% Inicializar una matriz para almacenar las configuraciones articulares interpoladas
q_interpolated = zeros(size(T_m, 3), 5);  

% Configuración articular inicial estimada
q0 = [0, 0, -90, 90, 0];

% Bucle sobre cada T(m) para obtener q(m) con la cinemática inversa
for i = 1:size(T_m, 3)
    % Aplicar la cinemática inversa para obtener la configuración articular q(m)
    q_interpolated(i, :) = cin_inv(R, T_m(:, :, i), q0, true); % 'true' para obtener la solución más cercana
    % Actualizar q0 para la próxima iteración
    q0 = q_interpolated(i, :);
end

%% Trajectoria por la pregunta 3.b

% figure (2)
% R.plot(q_interpolated(10,:));  % Posicion inicial
% for i=10:size(q_interpolated,1)
% R.plot(q_interpolated(i,:), 'trail', {'r', 'LineWidth', 2});  % Animacion de la trayectoria interpolada
% end


figure (2)
title('Trayectoria en el espacio articular despues de la interpolacion en el espacio cartesiano')
R.plot(q_interpolated(1,:), 'scale', 0.4);  % Posicion inicial
R.plot(q_interpolated, 'trail', {'r', 'LineWidth', 2});  % Animacion de la trayectoria interpolada

%% VELOCIDAD

%ESPACIO CARTESIANO 
dt = 0.1; % intervalo de tiempo
velocidad_espacio_cartesiano = diff(positions) / dt; % diferencias finitas para la velocidad
velocidad_espacio_cartesiano = [velocidad_espacio_cartesiano; velocidad_espacio_cartesiano(end, :)]; % para mantener el mismo tamaÃ±o

aceleracion_espacio_cartesiano = diff(velocidad_espacio_cartesiano) / dt; % diferencias finitas para la aceleraciÃ³n
aceleracion_espacio_cartesiano = [aceleracion_espacio_cartesiano; aceleracion_espacio_cartesiano(end, :)]; % mismo tamaÃ±o

%ESPACIO ARTICULAR
qd = diff(trayectoria_articular) / dt; % velocidad articular
qd = [qd; qd(end, :)];
qdd = diff(qd) / dt; % aceleración articular
qdd = [qdd; qdd(end, :)];

%% Visualización de los perfiles de posición, velocidad y aceleración 

%ESPACIO CARTESIANO
t = (0:size(positions, 1)-1) * dt; % eje de tiempos

figure (3);
subplot(3,1,1);
plot(t, positions);
title('Posicion en el espacio cartesiano');
xlabel('Tiempo [s]');
ylabel('Posicion [m]');
legend('x', 'y', 'z');

subplot(3,1,2);
plot(t, velocidad_espacio_cartesiano);
title('Velocidad en el espacio cartesiano');
xlabel('Tiempo [s]');
ylabel('Velocidad [m/s]');
legend('Vx', 'Vy', 'Vz');

subplot(3,1,3);
plot(t, aceleracion_espacio_cartesiano);
title('Aceleracion en el espacio cartesiano');
xlabel('Tiempo [s]');
ylabel('Aceleracion [m/s^2]');
legend('Ax', 'Ay', 'Az');

%ESPACIO ARTICULAR
 tiempos=(0:size(trayectoria_articular,1)-1)*dt
figure (4);
subplot(3,1,1);
plot(tiempos, trayectoria_articular*180/pi);
title('Posicion en el espacio articular');
xlabel('Tiempo [s]');
ylabel('Angulo [grados]');
legend('q1', 'q2', 'q3', 'q4', 'q5','q6');


subplot(3,1,2);
%figure
plot(tiempos, qd);
title('Velocidad en el espacio articular');
xlabel('Tiempo [s]');
ylabel('Velocidad [rad/s]');
legend('qd1', 'qd2', 'qd3', 'qd4', 'qd5','qd6');

subplot(3,1,3);
%figure
plot(tiempos, qdd);
title('aceleracion en el espacio articular');
xlabel('Tiempo [s]');
ylabel('Aceleracion [m/s^2]');
legend('Ax', 'Ay', 'Az');


%% CIN INV 
function Q = cin_inv(R, Tdato, q0, mejor)
if nargin == 3
    mejor = false;
elseif nargin ~= 4
    error('Argumentos incorrectos')
end

% Eliminación de offsets --------------------------------------------------
offsets = R.offset;
R.offset = zeros(1,5);

% Desacople de base y tool ------------------------------------------------
T =  invHomog(R.base.double) * Tdato * invHomog(R.tool.double);

% Inicializar solución:
qq = zeros(4,5);

% Cálculo de q1 -----------------------------------------------------------
% (expresión hallada por método matricial)
q1 = atan2(T(2,3), T(1,3));

qq(1,1) = q1;
qq(2,1) = q1;
qq(3,1) = q1;
qq(4,1) = q1;

% Cálculo de q2 -----------------------------------------------------------
% (expresión hallada por método matricial)
q2(1) = acos(T(3,3));
q2(2) = -q2(1);

% TD(3,3)/TD(2,3)*(-sin(q1)) = atan(q2) %otra forma de calcular q2
%atan2(T(3,3)*(-sin(q1)),T(2,3))

qq(1,2) = q2(1);
qq(2,2) = q2(1);
qq(3,2) = q2(2);
qq(4,2) = q2(2);

% Cálculo de q3 -----------------------------------------------------------
% conociendo q1 y q2 puedo calcular q3 y q4 planteando el problema clásico
% "codo arriba y codo abajo" entre el origen del sistema 2 y el origen del
% sistema 4.
%
% Calculo el origen del sistema 4 a partir de los datos de la matriz T.
% Referencio al sistema 2 y aplico solución geométrica.
p4 = T(1:3,4) - R.links(5).a * T(1:3,1);
for i=1:2
    T1 = R.links(1).A(q1).double;
    T2 = R.links(2).A(q2(i)).double;
    p = invHomog(T1 * T2) * [p4;1];
    
    B = atan2(p(2), p(1));
    r = sqrt(p(1)^2 + p(2)^2);
    a3 = R.links(3).a;
    a4 = R.links(4).a;
    G = acos((a3^2 + r^2 - a4^2) / (2 * r * a3));
    
    q3_1 = B - real(G);
    q3_2 = B + real(G);
    
    qq((i-1)*2+1,3) = q3_1;
    qq((i-1)*2+2,3) = q3_2;
end

% Cálculo de q4 -----------------------------------------------------------
% conociendo q1, q2 y q3 puedo calcular q4 continuando el problema clásico
for i=1:4
    T1 = R.links(1).A(qq(i,1)).double;
    T2 = R.links(2).A(qq(i,2)).double;
    T3 = R.links(3).A(qq(i,3)).double;
    p = invHomog(T1 * T2 * T3) * [p4;1];
    
    q4 = atan2(p(2), p(1));
    
    qq(i,4) = q4;
end

% Cálculo de q5 -----------------------------------------------------------
% conociendo q1, q2, q3 y q4 puedo calcular q5 obervando el punto final de
% T respecto del sistema 4
for i=1:4
    T1 = R.links(1).A(qq(i,1)).double;
    T2 = R.links(2).A(qq(i,2)).double;
    T3 = R.links(3).A(qq(i,3)).double;
    T4 = R.links(4).A(qq(i,4)).double;
    T45 = invHomog(T1 * T2 * T3 * T4) * T;
    
    q5 = atan2(T45(2,4), T45(1,4));
    
    qq(i,5) = q5;
end

% Offset ------------------------------------------------------------------
R.offset = offsets;
qq = qq - ones(4,1) * R.offset;

% Verificación total ------------------------------------------------------
%disp('Verificación total:')
Trpy = tr2rpy(Tdato);
Txyz = Tdato;
%fprintf('> T: [%f, %f, %f, %f, %f, %f]\n', Txyz(1,4), Txyz(2,4), Txyz(3,4), Trpy(1), Trpy(2), Trpy(3));

for i=1:4
    Taux = R.fkine(qq(i,:)).double;
    Tauxrpy = tr2rpy(Taux);
    %fprintf('> %d: [%f, %f, %f, %f, %f, %f]\n', i, Taux(1,4), Taux(2,4), Taux(3,4), Tauxrpy(1), Tauxrpy(2), Tauxrpy(3));
end

% Verificación de soluciones válidas
filas_a_eliminar = [];  % Lista para almacenar los índices de filas que vamos a eliminar

for i = 1:size(qq, 1)  % Iteración sobre cada fila de qq
    fuera_de_limites = false;  % Inicializamos una variable para indicar si está fuera de los lÃ­mites
    
    % Verificamos si cada articulación está dentro de sus límites 
    for j = 1:5  % Iteramos sobre las 5 articulaciones (columnas de qq)
        if qq(i, j) < R.qlim(j, 1) || qq(i, j) > R.qlim(j, 2)
            fuera_de_limites = true;  % Marcamos que estÃ¡ fuera de límites
            break;  % Salimos del bucle interno si una articulación está fuera de lÃ­mites
        end
    end
    
    % Si está fuera de los lí­mites, añadimos el índice de la fila a la lista
    if fuera_de_limites
        filas_a_eliminar = [filas_a_eliminar; i];
    end
end

% Para eliminar las filas fuera de lÃ­mites de qq
qq(filas_a_eliminar, :) = [];

% Devolución --------------------------------------------------------------
% la mejor respuesta es la que tiene menor distancia a la anterior 
% Selección de la solución más cercana a q0 --------------------------------
if mejor
    if ~isempty(qq)
        % Calculamos la distancia entre cada solución en qq y q0
        distancias = vecnorm(qq - q0, 2, 2);  % Norma euclidiana para cada fila
    
        % Encontramos el índice de la solución más cercana
        [~, indice_minimo] = min(distancias);
    
        % Seleccionamos la solución más cercana
        Q = qq(indice_minimo, :);
    else
        % Si no hay soluciones válidas, devuelve un error 
        error('No hay soluciones vÃ¡lidas que cumplan los lÃ­mites.');
    end      
else
        Q = qq;
end

end
% Inversa de matriz homogénea
function iT = invHomog(T)
iT = eye(4);
iT(1:3, 1:3) = T(1:3, 1:3)';
iT(1:3, 4) = - iT(1:3, 1:3) * T(1:3, 4);
end

