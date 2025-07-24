clear 
clc
close all 

robotatalante; %llamado al archivo robotatalante.m

% vector de posiciones articulares que se desee analizar.
q = [10, -20, 45, -45, 0]*pi/180; %problema a analizar

% vector de booleanos 
sistemas = [1, 1, 1, 1, 1, 1]; 

% Ploteo del robot en la posición definida
figure;
R.plot(q, 'scale', 0.5, 'jointdiam', 0.5, 'notiles');

% bucle recorra todos los sistemas posibles del robot, y grafíquelos de acuerdo al vector definido
hold on;

% Graficar el sistema de referencia de la base
trplot(R.base, 'frame', '0', 'rgb', 'length', 0.1);

% Inicializar transformación acumulada (empieza desde la base)
T_acumulada = R.base;

% Bucle para graficar los sistemas de referencia en cada articulación
for i = 1:length(sistemas)
    if sistemas(i) == 1
        % Obtener la transformación de la articulación i
        T_acumulada = T_acumulada * R.A(i, q);  % Acumular la transformación
        trplot(T_acumulada, 'frame', num2str(i), 'rgb', 'length', 0.1);  % Graficar el sistema de referencia
    end
end

hold off;
% Nota: Appliquer des transformations de base et d'outil si nécessaire
%R.base = transl(0.2, 0.2, 0) * trotx(pi/6); % Exemple de transformation de base
%R.tool = transl(0.1, 0, 0.1) * trotz(pi/4); % Exemple de transformation de l'outil

