function Q = cin_inv(R, Tdato, q0, mejor)
%% CIN_INV 
if nargin == 3
    mejor = false;
elseif nargin ~= 4
    error('Argumentos incorrectos')
end
% Eliminación de offsets --------------------------------------------------
%offsets = R.offset;
%R.offset = [0 -pi/2 pi/2 0 -pi/2]; % Offset inicial
R.tool = transl(0, 0, 0);
R.base = transl(0, 0, 0) * trotx(-pi/2);
% Desacople de base y tool ------------------------------------------------
T =  invHomog(R.base) * Tdato * invHomog(R.tool);
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
R.offset = [0 -pi/2 pi/2 0 -pi/2];
qq = qq - ones(4,1) * R.offset;
% Verificación total ------------------------------------------------------
%disp('Verificación total:')
Trpy = tr2rpy(Tdato);
Txyz = Tdato.double;
%fprintf('> T: [%f, %f, %f, %f, %f, %f]\n', Txyz(1,4), Txyz(2,4), Txyz(3,4), Trpy(1), Trpy(2), Trpy(3));
for i=1:4
    Taux = R.fkine(qq(i,:)).double;
    Tauxrpy = tr2rpy(Taux);
    %fprintf('> %d: [%f, %f, %f, %f, %f, %f]\n', i, Taux(1,4), Taux(2,4), Taux(3,4), Tauxrpy(1), Tauxrpy(2), Tauxrpy(3));
end
% Verificación de soluciones válidas
filas_a_eliminar = [];  % Lista para almacenar los índices de filas que vamos a eliminar
for i = 1:size(qq, 1)  % Iteración sobre cada fila de qq
    fuera_de_limites = false;  % Inicializamos una variable para indicar si está fuera de los límites
    
    % Verificamos si cada articulación está dentro de sus límites 
    for j = 1:5  % Iteramos sobre las 5 articulaciones (columnas de qq)
        if qq(i, j) < R.qlim(j, 1) || qq(i, j) > R.qlim(j, 2)
            fuera_de_limites = true;  % Marcamos que está fuera de límites
            break;  % Salimos del bucle interno si una articulación está fuera de límites
        end
    end
    
    % Si está fuera de los límites, añadimos el índice de la fila a la lista
    if fuera_de_limites
        filas_a_eliminar = [filas_a_eliminar; i];
    end
end
% Para eliminar las filas fuera de límites de qq
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
        error('No hay soluciones válidas que cumplan los límites.');
    end      
else
        Q = qq;
end


end
%=========================================================================%
% Inversa de matriz homogénea
%%
function iT = invHomog(T)
    % Verifica que T sea una matriz homogénea 4x4
    if ~isequal(size(T), [4, 4])
        error('La matriz de entrada debe ser homogénea de tamaño 4x4.');
    end
    % Calcula la inversa de una matriz de transformación homogénea
    iT = eye(4);
    iT(1:3, 1:3) = T(1:3, 1:3)';
    iT(1:3, 4) = -iT(1:3, 1:3) * T(1:3, 4);
end
