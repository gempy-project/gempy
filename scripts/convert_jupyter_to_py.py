import os
import pypandoc as pdoc
import json

# %%


def convert_ipynb_to_gallery(nb, new_file):
    python_file = ""

    nb_dict = json.load(open(nb, encoding="utf8", errors='ignore'))
    cells = nb_dict['cells']

    for i, cell in enumerate(cells):
        if i == 0:
            if cell['cell_type'] != 'markdown':
                rst_source = os.path.basename(file_name[:-5])
                rst_source = bytes(rst_source, 'utf-8').decode('utf-8', 'ignore')
                python_file = '"""\n' + rst_source + '\n"""'
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source

            else:
                b = cell['source']
                print(b)
                a = bytes(cell['source'][0], 'utf-8').decode('utf-8', 'ignore')
                print(a)
                md_source = ''.join(a)
                rst_source = pdoc.convert_text(md_source, 'rst', 'md')
                print(rst_source)
                rst_source = bytes(rst_source, 'utf-8').decode('utf-8', 'ignore')
                python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell['cell_type'] == 'markdown':
                md_source = ''.join(cell['source'])
                rst_source = pdoc.convert_text(md_source, 'rst', 'md')
                rst_source = rst_source.encode().decode('utf-8', 'ignore')
                commented_source = '\n'.join(['# ' + x for x in
                                              rst_source.split('\n')])
                #python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                #    commented_source

                python_file = python_file + '\n\n\n' + '# %%' + '\n' + \
                              commented_source

            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + '# %% \n' + source

    python_file = python_file.replace("\n%", "\n# %")
    open(new_file, 'w', newline='',  errors='ignore').write(python_file)

#%%
directory = os.getcwd()+'/../notebooks/integrations/'
#directory = '/mnt/i/PycharmProjects/gempy_notebooks/notebooks/Probabilistic Modeling'

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.ipynb'):
            if file.find('checkpoint') != -1:
                continue

           # print(file.find('checkpoint'), file,  root)
            file_name = root+'/'+file
            print(os.path.basename(file_name[:-5]))
            nb = root+'/'+file #jupytext.read(root+'/'+file)
            if True:
                new_dir = root.replace('notebooks', 'examples_')
            else:
                pass
            new_file = new_dir+'/'+file.replace('.ipynb', '.py')
            try:
                os.makedirs(new_dir)
            except:
                pass
            #jupytext.write(nb, new_file, fmt='sphinx')
            convert_ipynb_to_gallery(nb, new_file)


