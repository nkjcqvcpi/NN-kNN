"""T3 tools write generated output under results/t3/<stage>/ and refuse reports/ and docs/ (O2/O3)."""
import importlib
from pathlib import Path
import sys

import pytest

from model.t3 import outputs

ROOT=Path(outputs.__file__).resolve().parents[2]


def test_the_default_output_is_a_fresh_path_under_results():
    assert outputs.RESULTS==ROOT/'results'/'t3'
    path=outputs.output_path(None,'s0_host_setup','t3-host-smoke','.json')
    assert path.parent==ROOT/'results'/'t3'/'s0_host_setup' and path.name.startswith('t3-host-smoke-') and path.suffix=='.json'
    folder=outputs.output_path(None,'s1_single_hop_retrieval','t3-public-host')
    assert folder.parent==ROOT/'results'/'t3'/'s1_single_hop_retrieval' and not folder.exists()


def test_an_explicit_output_is_kept_unless_it_is_inside_reports_or_docs(tmp_path):
    assert outputs.output_path(tmp_path/'x','s0_host_setup','n')==tmp_path/'x'
    assert outputs.output_path(ROOT/'results'/'t3'/'x','s0_host_setup','n')==ROOT/'results'/'t3'/'x'
    assert outputs.output_path(ROOT/'reportsx','s0_host_setup','n')==ROOT/'reportsx'
    for path in (ROOT/'reports',ROOT/'reports'/'a'/'b.json',ROOT/'docs'/'t3'):
        with pytest.raises(ValueError,match='only written reports'):outputs.output_path(path,'s0_host_setup','n')


TOOLS={  # tool -> its output option and the other required options (argument parsing stops before they are used)
    't3_host_smoke':('--out',['--model-dir','m','--files-manifest','f','--expected-revision','r']),
    't3_prepare_host':('--manifest',['--revision','0'*40,'--model-dir','m']),
    't3_prompt_pilot':('--out',['--model-dir','m','--files-manifest','f','--runtime-library-bin','b']),
    't3_public_data_audit':('--output',['--data-root','d']),
    't3_public_host_pilot':('--output',['--model-dir','m','--host-manifest','h','--runtime-library-bin','b','--data-root','d',
                                        '--sample-manifest','s','--metric-checkpoint','c']),
    't3_internal_host_pilot':('--output',[f'--{n}' if i%2==0 else 'x' for n in ('source-run','source-verification','model-dir',
        'runtime-library-bin','data-root','metric-checkpoint','format-library') for i in (0,1)]),
    't3_public_set_utility':('--output',[f'--{n}' if i%2==0 else 'x' for n in ('input-run','input-verification','model-dir',
        'host-manifest','data-root','runtime-library-bin','format-library') for i in (0,1)]),
    't3_train_need_queries':('--output',[f'--{n}' if i%2==0 else 'x' for n in ('reference-run','reference-verification','data-root',
        'samples','model-dir','runtime-library-bin','format-library') for i in (0,1)]),
}


@pytest.mark.parametrize('tool',sorted(TOOLS))
@pytest.mark.parametrize('folder',['reports','docs'])
def test_every_host_tool_refuses_an_output_inside_reports_or_docs(tool,folder,monkeypatch,capsys):
    option,required=TOOLS[tool]
    module=importlib.import_module(f'tools.{tool}')
    target=ROOT/folder/'t3-generated-test'
    monkeypatch.setattr(sys,'argv',[tool,*required,option,str(target)])
    with pytest.raises(SystemExit) as exc:module.main()
    assert exc.value.code==2 and 'only written reports' in capsys.readouterr().err and not target.exists()


@pytest.mark.parametrize('tool',sorted(TOOLS))
def test_no_host_tool_requires_its_output_option(tool,monkeypatch):
    option,required=TOOLS[tool]
    module=importlib.import_module(f'tools.{tool}')
    seen={}
    def stop(path,stage,name,suffix=''):
        seen.update(path=path,stage=stage,name=name);raise SystemExit(0)
    monkeypatch.setattr(module,'output_path',stop)
    monkeypatch.setattr(sys,'argv',[tool,*required])
    with pytest.raises(SystemExit) as exc:module.main()
    assert exc.value.code==0 and seen['path'] is None and seen['stage'] in ('s0_host_setup','s1_single_hop_retrieval','s1_model_integration')
