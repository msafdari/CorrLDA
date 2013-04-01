clear ff A y i
listing = dir('.');
A = [];
for i=5:numel(listing)
A(i-4,:) = dlmread(listing(i).name);
end
[y, i] = max(A);
for j=1:size(A,1)
ff(j) = numel(find(i==j));
end
plot(ff)