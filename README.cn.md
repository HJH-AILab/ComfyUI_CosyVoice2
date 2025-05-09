# ComfyUI_CosyVoice2
A wrapper of <a href="https://github.com/FunAudioLLM/CosyVoice/">CosyVoice2</a>'s ComfyUI custom_nodes

#安装
1.
<code>
    git clone https://github.com/HJH-AILab/ComfyUI_CosyVoice2.git
    cd ComfyUI_CosyVoice2
    git clone ttps://github.com/FunAudioLLM/CosyVoice.git
</code>

2.
<p>
    按照<a href="https://github.com/FunAudioLLM/CosyVoice/">CosyVoice2</a>的安装说明安装CosyVoice2,并下载模型.
    *提示: 安装原项目建议删除requirements.txt中的版本号安装, 经测试可以在torch2.7.0+cuda128下运行.

    *提示: 原项目依赖diffusers==0.29.0,会造成comfyui中大部节点报错(依赖更新的diffusers库的节点), 经测试可安装0.33.1版本后修改一行代码正常运行
    修改: CosyVoice\cosyvoice\flow\decoder.py line:230 为:
    <code>
    super(CausalAttention, self).__init__(query_dim=query_dim, cross_attention_dim=cross_attention_dim, heads=heads, dim_head=dim_head, dropout=dropout, bias=bias, upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
                                              cross_attention_norm=cross_attention_norm, cross_attention_norm_num_groups=cross_attention_norm_num_groups, qk_norm=qk_norm, added_kv_proj_dim=added_kv_proj_dim, norm_num_groups=norm_num_groups,
                                              spatial_norm_dim=spatial_norm_dim, out_bias=out_bias, scale_qk=scale_qk, only_cross_attention=only_cross_attention, eps=eps, rescale_output_factor=rescale_output_factor, residual_connection=residual_connection,
                                              _from_deprecated_attn_block=_from_deprecated_attn_block, processor=processor, out_dim=out_dim)
    </code>
    即将参数改为显示输入即可兼容0.33.1
</p>

3.
<p>
    配置Comfyui extra_model_paths.yaml
    <code>
    cosyvoice: &lt;your path to CosyVoice2's models>
    </code>
</p>