import jupytext, os

directory = os.getcwd()+'/notebooks'

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.ipynb'):
            if file.find('checkpoint') != -1:
                continue
            print(file.find('checkpoint'), file,  root)
            file_name = root+'/'+file
            nb = jupytext.read(root+'/'+file)
            new_dir = root.replace('notebooks', 'examples')
            new_file = new_dir+'/'+file.replace('.ipynb', '.py')
            try:
                os.makedirs(new_dir)
            except WindowsError as err:
                print(err)
            jupytext.write(nb, new_file, fmt='py:sphinx')



