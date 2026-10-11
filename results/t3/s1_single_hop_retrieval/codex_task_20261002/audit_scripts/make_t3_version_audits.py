from pathlib import Path
work = Path(__file__).resolve().parent
for old,new in [('audit_t3_prompt_pilot','audit_t3_version_host'),('audit_t3_direct_removal','audit_t3_version_removal')]:
    text = (work/f'{old}.py').read_text(encoding='utf-8')
    for source in ('t3_prompt_pilot_staged_schema','t3_prompt_direct_removal'):
        text = text.replace(source,'t3_version_binding')
    text = text.replace('outputs/t3-prompt-pilot-20261004-staged-schema','outputs/t3-version-binding-host-20261005')
    text = text.replace('outputs/t3-prompt-pilot-20261004-direct-removal','outputs/t3-version-binding-host-20261005')
    text = text.replace('outputs/neural-cbr-t3-prompt-20261004-verification.json','outputs/neural-cbr-t3-version-host-verification.json')
    text = text.replace('outputs/neural-cbr-t3-direct-removal-20261004-verification.json','outputs/neural-cbr-t3-version-removal-verification.json')
    if old == 'audit_t3_prompt_pilot':
        text = text.replace('assert replay["evidence"] == event["evidence"]', '''assert replay["evidence"] == event["evidence"]
        for key in ("model_state_sha256", "encoder_state_sha256", "encoder_verification", "bank_version"):
            assert replay["audit"][key] == audit[key]
        assert audit["encoder_verification"] == "declared_only" and audit["encoder_state_sha256"] is None''')
    (work/f'{new}.py').write_text(text,encoding='utf-8')
print('version audit helpers ready')
