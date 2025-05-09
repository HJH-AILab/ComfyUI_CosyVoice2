# ComfyUI_CosyVoice2
A wrapper of [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice/ "CosyVoice2")'s ComfyUI custom_nodes  

<p style="text-align:center;"><a href="README.md">English</a> | <span>中文</span></p>

# 安装
1.   
>git clone <https://github.com/HJH-AILab/ComfyUI_CosyVoice2.git>  
>cd ComfyUI_CosyVoice2  
>git clone <https://github.com/FunAudioLLM/CosyVoice.git>  

2.  
>按照[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice/ "CosyVoice2")的安装说明安装CosyVoice2,并下载模型.  

>*提示: 安装原项目建议删除requirements.txt中的版本号安装, 经测试可以在torch2.7.0+cuda128下运行.  

>*提示: 原项目依赖diffusers==0.29.0,会造成comfyui中大部节点报错(依赖更新的diffusers库的节点), 经测试可安装0.33.1版本后修改一行代码正常运行.  
>>修改: CosyVoice\cosyvoice\flow\decoder.py line:230 为:  
>>``super(CausalAttention, self).__init__(query_dim=query_dim, cross_attention_dim=cross_attention_dim, heads=heads, dim_head=dim_head, dropout=dropout, bias=bias, upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
                                                    cross_attention_norm=cross_attention_norm, cross_attention_norm_num_groups=cross_attention_norm_num_groups, qk_norm=qk_norm, added_kv_proj_dim=added_kv_proj_dim, norm_num_groups=norm_num_groups,
                                                    spatial_norm_dim=spatial_norm_dim, out_bias=out_bias, scale_qk=scale_qk, only_cross_attention=only_cross_attention, eps=eps, rescale_output_factor=rescale_output_factor, residual_connection=residual_connection,
                                                    _from_deprecated_attn_block=_from_deprecated_attn_block, processor=processor, out_dim=out_dim)``  
>>即将参数改为显示输入即可兼容0.33.1  

3.  
配置Comfyui extra_model_paths.yaml  
>添加:  
>cosyvoice: &lt;your path to CosyVoice2's models>  

# 工作流
![工作流](exsample/workflow.png "workflow")