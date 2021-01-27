clear all;

n = 5;
k = 4;
genpoly = bchgenpoly(n, k);
int32gen = flip(genpoly.x);
intgen = zeros(1, length(int32gen));
for i = 1:length(int32gen)
    intgen(i) = int32gen(i);
end
totalpoly = zeros(1, n + 1);
totalpoly(1) = 1;
totalpoly(n + 1) = 1;
[revqpoly, rpoly] = gfdeconv(totalpoly, intgen);
qpoly = flip(revqpoly);

fileID = fopen('GenAndPar.txt', 'w');
fprintf(fileID, ' n = %d \n k = %d \n Generator matrix row: \n', n, k);
for i = 1:length(intgen)
    fprintf(fileID, '%d ', intgen(i));
end
fprintf(fileID, '\n Parity matrix row: \n');
for i = 1:length(qpoly)
    fprintf(fileID, '%d ', qpoly(i));
end
fclose(fileID);