load 'C:\Users\Daniel\Desktop\gt.mat';
for i = 1 : length(txt)
	c = txt{i}';
	c = c(:);
	txt(i) = { (c(c~=' ' & c~=newline))' };
end
file = fopen('synth_data.csv', 'w');
for i = 1 : length(imnames)
    if imnames{i}(1) ~= '8' 
        break;
    end
	fprintf(file, '%s,|%s|,', imnames{i}, txt{i});
	fprintf(file, "%f,", charBB{i}(:));
	fprintf(file, "\n");
end
fclose(file);