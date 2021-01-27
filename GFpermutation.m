clear all;

m = 6;
n = 2^m;
A = gf((0:(n - 1)), m);
for i = 4:n
    A(i) = A(i - 1) * gf(2, m);
end

intA = A.x;
inverseA = intA;
for i = 1:n
    inverseA(intA(i) + 1) = i - 1;
end

C = gf(zeros(n, n), m);
for i = 1:n
    for j = 1:n
        C(i, j) = A(j) + gf(i - 1, m);
    end
end
intC = C.x;

for i = 1:n
    for j = 1:n
        intC(i, j) = inverseA(intC(i, j) + 1);
    end
end

fileID = fopen('GFpermutation.txt', 'w');
fprintf(fileID, ' m = %d \n n = %d \n Permutation matrix is \n', m, n);
for i = 1:n
    for j= 1:n
        fprintf(fileID, '%d ', intC(i,j));
    end
    fprintf(fileID, '\n');
end
fclose(fileID);