# Fast OLMo-core converter for chankhavu/smolmo-sft-v2-seqlen64k (single + multi node).
# Base has the open-instruct converter + Olmo tokenizer deps. We overlay two patched files
# (tokenization/masking UNCHANGED — proven byte-identical) + driver/entrypoint/inspect.
#   * dataset_transformation.py : count_tokens num_proc + direct parquet glob/list input
#   * numpy_dataset_conversion.py: SMOLMO_NO_SHUFFLE env-gate (order-only; for deterministic multi-node)
FROM open-instruct-dataprep:0.1.0
ENV OPEN_INSTRUCT=/workspace/open-instruct
COPY dataset_transformation.PATCHED.py /workspace/open-instruct/open_instruct/dataset_transformation.py
COPY numpy_dataset_conversion.PATCHED.py /workspace/open-instruct/open_instruct/numpy_dataset_conversion.py
COPY entrypoint.sh /opt/convert/entrypoint.sh
COPY convert_node.py /opt/convert/convert_node.py
COPY inspect_olmocore.py /opt/convert/inspect_olmocore.py
RUN chmod +x /opt/convert/entrypoint.sh
WORKDIR /out
ENTRYPOINT ["/opt/convert/entrypoint.sh"]
