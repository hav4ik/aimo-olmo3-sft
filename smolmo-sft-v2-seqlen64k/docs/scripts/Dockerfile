# Self-contained image to convert chankhavu/smolmo-sft-v2-seqlen64k -> OLMo-core pretokenized.
# Base already contains the open-instruct converter + Olmo tokenizer deps.
FROM open-instruct-dataprep:0.1.0
ENV OPEN_INSTRUCT=/workspace/open-instruct
COPY convert_hf_to_olmocore.py /opt/convert/convert_hf_to_olmocore.py
COPY entrypoint.sh /opt/convert/entrypoint.sh
RUN chmod +x /opt/convert/entrypoint.sh
WORKDIR /out
ENTRYPOINT ["/opt/convert/entrypoint.sh"]
