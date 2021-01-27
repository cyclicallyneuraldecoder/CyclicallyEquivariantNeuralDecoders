clear all;

m = 8;
r = 3;
n = 2^m;
k = 0;
for i = 0:r
    k = k + nchoosek(m, i);
end

A = gf((1:(n - 1)), m);
for i = 3:n
    A(i) = A(i - 1) * gf(2, m);
end
poly = flip(minpol(A(2)));
int32poly = poly.x;
intpoly = zeros(1, length(int32poly));
for i = 1:length(int32poly)
    intpoly(i) = int32poly(i);
end
for i = 2:(n - 1)
    wti = 0;
    temp = 1;
    for j = 1:m
        if bitand(i, temp) ~= 0
            wti = wti + 1;
        end
        temp = temp * 2;
    end
    if wti >= (m - r)
        continue;
    end    
    temppoly = flip(minpol(A(i + 1)));
    int32temppoly = temppoly.x;
    inttemppoly = zeros(1, length(int32temppoly));
    for j = 1:length(int32temppoly)
        inttemppoly(j) = int32temppoly(j);
    end
    [quot, remd] = gfdeconv(intpoly, inttemppoly);
    if remd == 0
        continue;
    end
    intpoly = gfconv(intpoly, inttemppoly);
end

totalpoly = zeros(1, n);
totalpoly(1) = 1;
totalpoly(n) = 1;
[revqpoly, rpoly] = gfdeconv(totalpoly, intpoly);
qpoly = flip(revqpoly);

fileID = fopen('RMGenAndPar.txt', 'w');
fprintf(fileID, 'Punctured Reed-Muller codes parameters are \n m = %d \n r = %d \n n = %d \n k = %d \n Generator matrix row: \n', m, r, n - 1, k);
for i = 1:length(intpoly)
    fprintf(fileID, '%d ', intpoly(i));
end
fprintf(fileID, '\n Parity matrix row: \n');
for i = 1:length(qpoly)
    fprintf(fileID, '%d ', qpoly(i));
end
fclose(fileID);