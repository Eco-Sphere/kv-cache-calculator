const MODELS = [
  {family:"MiniMax",id:"minimax-m3",label:"MiniMax M3",type:"minimax_msa",layers:60,fullLayers:3,sparseLayers:57,heads:4,dim:128,indexDim:128,indexHeads:4,indexBlockSize:128,indexTopkBlocks:16,indexLocalBlocks:1,mtpModules:7,nextNLayers:1,max:1048576,fixedIndexerPrecision:2,source:"MiniMaxAI/MiniMax-M3"}
];
const $=id=>document.getElementById(id);const families=[...new Set(MODELS.map(m=>m.family))];
families.forEach(f=>$("family").add(new Option(f,f)));
function modelsForFamily(){return MODELS.filter(m=>m.family===$("family").value)}
function fillModels(){const previous=$("model").value;$("model").replaceChildren();modelsForFamily().forEach(m=>$("model").add(new Option(m.label,m.id)));if(modelsForFamily().some(m=>m.id===previous))$("model").value=previous;syncModel()}
function selected(){return MODELS.find(m=>m.id===$("model").value)||modelsForFamily()[0]}
function formatBytes(bytes){const u=["B","KiB","MiB","GiB","TiB"];let n=bytes,i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return`${n.toFixed(n>=100?1:n>=10?2:3)} ${u[i]}`}
function syncModel(){const m=selected();$("tokens").value=Math.min(1024,m.max);const hasIndexer=m.type==="mla"||m.type==="minimax_msa";$("indexer-control").hidden=!hasIndexer;$("indexer").disabled=Boolean(m.fixedIndexerPrecision);if(m.fixedIndexerPrecision)$("indexer").value=String(m.fixedIndexerPrecision);$("draft-control").hidden=!m.draft;$("draft").checked=false;calculate()}
function renderFormula(rows){$("formula-list").replaceChildren(...rows.map(([name,expression])=>{const row=document.createElement("div");row.className="formula-row";row.innerHTML=`<span class="formula-name">${name}</span><span class="formula-equals">=</span><span class="formula-expression">${expression}</span>`;return row}))}
function renderBreakdown(rows){$("breakdown").replaceChildren(...rows.map(([name,value,help])=>{const row=document.createElement("div");row.className="breakdown-row";const label=document.createElement("span");label.textContent=name;if(help){const button=document.createElement("button");button.type="button";button.className="kv-help";button.textContent="?";button.setAttribute("aria-label",help);button.dataset.tooltip=help;label.append(button)}const strong=document.createElement("strong");strong.textContent=typeof value==="number"?value.toLocaleString():value;row.append(label,strong);return row}))}
function calculate(){const m=selected();const tokens=Math.max(1,+$("tokens").value||1);const seq=Math.max(1,+$("sequences").value||1);const tp=Math.max(1,Math.floor(+$("tensor-parallel").value||1));const p=+$("precision").value;let kvPerToken,indexerPerToken=0,label,note,rows;
  if(m.type==="minimax_msa"){const ip=m.fixedIndexerPrecision||+$("indexer").value;kvPerToken=m.layers*2*m.heads*m.dim*p;indexerPerToken=m.sparseLayers*m.indexDim*ip;label="MiniMax MSA sparse attention";rows=[["kv_bytes","tokens × sequences × layers × 2 × num_key_value_heads × head_dim × kv_precision_bytes"],["indexer_bytes","tokens × sequences × sparse_attention_layers × index_head_dim × indexer_precision_bytes"],["total_bytes","kv_bytes + indexer_bytes"]];note="MiniMax Sparse Attention (MSA) uses a lightweight indexer to select relevant KV blocks, with a separate key-only indexer cache for block selection."}
  else if(m.type==="mla"){const ip=+$("indexer").value;kvPerToken=m.layers*(m.rank+m.rope)*p;indexerPerToken=m.layers*m.indexDim*ip;label="MLA latent KV + indexer";rows=[["kv_bytes","tokens × sequences × layers × (kv_lora_rank + qk_rope_head_dim) × kv_precision_bytes"],["indexer_bytes","tokens × sequences × layers × index_head_dim × indexer_precision_bytes"],["total_bytes","kv_bytes + indexer_bytes"]];note="MLA latent cache plus sparse indexer cache. Indexer precision is selected independently."}
  else{const layers=m.layers+($("draft").checked?(m.draft||0):0);kvPerToken=2*layers*m.heads*m.dim*p;label="Standard GQA / MQA KV cache";rows=[["total_bytes","tokens × sequences × active_layers × 2 × num_key_value_heads × head_dim × precision_bytes"]];note="Stores both key and value tensors for every cached token and sequence."}
  const perToken=kvPerToken+indexerPerToken,one=perToken*tokens,kvTotal=kvPerToken*tokens*seq,indexerTotal=indexerPerToken*tokens*seq,total=one*seq;$("total-gib").textContent=`${(total/1024**3).toFixed(5)} GiB`;$("total-gb").textContent=`= ${(total/1e9).toFixed(5)} GB`;$("per-device").textContent=formatBytes(total/tp);
  if(indexerPerToken){$("metric-one-label").textContent="KV cache size";$("metric-one").textContent=formatBytes(kvTotal);$("metric-two-label").textContent="Indexer cache size";$("metric-two").textContent=formatBytes(indexerTotal);$("metric-three-label").textContent="Per token size";$("metric-three").textContent=formatBytes(perToken)}else{$("metric-one-label").textContent="Per sequence";$("metric-one").textContent=formatBytes(one);$("metric-two-label").textContent="Per token";$("metric-two").textContent=formatBytes(perToken);$("metric-three-label").textContent="Bytes per element";$("metric-three").textContent=`${p} byte${p===1?"":"s"}`}
  $("formula-label").textContent=label;renderFormula(rows);$("cache-note").textContent=note;
  if(m.type==="minimax_msa")renderBreakdown([
    ["Main layers",m.layers],
    ["Full-attention layers",m.fullLayers,"Dense/full-attention layers without the MSA indexer branch."],
    ["Sparse-attention layers",m.sparseLayers,"MSA layers that add a key-only indexer side cache."],
    ["KV elements per token",m.layers*2*m.heads*m.dim,"Main K/V elements per token before applying KV precision."],
    ["Indexer elements per token",m.sparseLayers*m.indexDim,"MSA index-key elements per token before applying indexer precision."],
    ["Index heads",m.indexHeads,"Indexer query heads used for scoring selected KV blocks; the stored index-key cache is key-only."],
    ["Index block size",m.indexBlockSize,"Number of tokens per sparse-selection block. This is not a sliding-window cache cap."],
    ["Top-k blocks",m.indexTopkBlocks,"Sparse blocks selected by the indexer for each query."],
    ["Local blocks",m.indexLocalBlocks,"Recent local blocks always visible to the sparse attention path."],
    ["MTP modules not included",m.mtpModules,"The public MiniMax M3 checkpoint/config exposes MTP fields, but bundled MTP weights are not modeled in the base serving path."],
    ["Next-N layers not included",m.nextNLayers,"Config field retained for traceability; draft KV is not included for MiniMax M3."],
    ["Model fields","num_hidden_layers=60, full_attention_layers=3, sparse_attention_layers=57, num_key_value_heads=4, head_dim=128, index_head_dim=128, index_n_heads=4, index_block_size=128, index_topk_blocks=16, index_local_blocks=1, indexer_fixed_precision_id=bf16_fp16"],
    ["KV precision bytes",p],
    ["Indexer precision bytes",m.fixedIndexerPrecision||+$("indexer").value],
    ["Tensor parallel size",tp],
    ["Per-device cache size",formatBytes(total/tp)]
  ]);else renderBreakdown([["Model layers",m.layers],["KV cache precision",$("precision").selectedOptions[0].textContent],["Workload",`${tokens.toLocaleString()} tokens × ${seq.toLocaleString()} sequences`]]);
  const sourceUrl=`https://huggingface.co/${m.source}/raw/main/config.json`;$("source").textContent=sourceUrl;$("source-link").href=sourceUrl}
$("family").addEventListener("change",fillModels);$("model").addEventListener("change",syncModel);$("cache-form").addEventListener("input",calculate);fillModels();
