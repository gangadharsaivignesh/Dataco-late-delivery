"""Equivalent to `jupyter nbconvert --to notebook --execute --inplace` for machines without Jupyter.
Runs notebooks/Late_Delivery_Analysis.ipynb cell by cell and stores the outputs in the file. Run from the repo root.
Usage: python scripts/exec_notebook.py"""
import json, ast, io, sys, base64, contextlib, traceback
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P='notebooks/Late_Delivery_Analysis.ipynb'
nb=json.load(open(P)); ns={'__name__':'__main__'}
cnt=0
for c in nb['cells']:
    if c['cell_type']!='code': continue
    cnt+=1; src=''.join(c['source']); outs=[]; buf=io.StringIO(); figs=[]
    def show(*a,**k):
        for n in plt.get_fignums():
            f=plt.figure(n); b=io.BytesIO(); f.savefig(b,format='png',dpi=100,bbox_inches='tight'); figs.append(b.getvalue())
        plt.close('all')
    plt.show=show
    last=None
    try:
        tree=ast.parse(src)
        if tree.body and isinstance(tree.body[-1],ast.Expr):
            last=ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf):
            exec(compile(tree,'cell','exec'),ns)
            val=eval(compile(last,'cell','eval'),ns) if last else None
    except Exception:
        outs.append({'output_type':'error','ename':'Error','evalue':'','traceback':traceback.format_exc().splitlines()}); val=None
        print('ERROR in cell',cnt,traceback.format_exc()[-400:],file=sys.stderr)
    show()
    if buf.getvalue(): outs.append({'output_type':'stream','name':'stdout','text':buf.getvalue().splitlines(True)})
    for f in figs: outs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(f).decode(),'text/plain':['<Figure>']}})
    if val is not None:
        d={'text/plain':repr(val).splitlines(True)}
        if hasattr(val,'to_html'): d['text/html']=val.to_html(max_rows=30).splitlines(True)
        outs.append({'output_type':'execute_result','metadata':{},'execution_count':cnt,'data':d})
    c['outputs']=outs; c['execution_count']=cnt
json.dump(nb,open(P,'w'),indent=1); print('executed',cnt,'code cells')
