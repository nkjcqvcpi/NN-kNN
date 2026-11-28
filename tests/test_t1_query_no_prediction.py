import pytest
import torch
from model.t1.query_aware import query_adapter_type
from tools.t1_query_no_prediction import prediction_only_probe, ratio_summary


@pytest.mark.parametrize("kind",["classification","regression"])
def test_probe_changes_only_retrieval_addition(kind):
    torch.manual_seed(15)
    dim=3 if kind=="classification" else 1
    adapter=query_adapter_type(kind,"query_aware_no_prediction")(2,dim,hidden_dims=(8,4))
    values=dict(query=torch.randn(7,2),aggregate=torch.randn(7,2),p0=torch.randn(7,dim))
    if kind=="classification": values["p0"]=values["p0"].softmax(1)
    before={k:v.clone() for k,v in values.items()}
    target=torch.arange(7)%dim if kind=="classification" else torch.randn(7)
    row,output=prediction_only_probe(adapter,kind,values,target)
    assert row["intervention"]=="rotate_prediction_only"
    assert row["residual_max_change"]<1e-6
    assert row["translation_max_error"]<1e-6
    assert output.shape==values["p0"].shape
    assert all(torch.equal(v,before[k]) for k,v in values.items())
    if kind=="regression": assert row["sensitivity_ratio"]==pytest.approx(1.)


def test_comparison_retains_zero_errors_and_excludes_only_their_ratio():
    result=ratio_summary([(dict(task="a",test_error=0.),dict(test_error=.2)),
                          (dict(task="b",test_error=.1),dict(test_error=.05)),
                          (dict(task="c",test_error=.3),dict(test_error=.3))])
    assert (result["wins"],result["ties"],result["losses"])==(1,1,1)
    assert result["zero_exclusions"]==["a"]
    assert result["gmr"]==pytest.approx(2**.5)
