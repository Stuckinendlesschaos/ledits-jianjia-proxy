from flask import Flask
from flask import request
from flask import send_file, jsonify


from  flask_cors import CORS
from gradio_client import Client

from constants import *
import io,base64

class Entity():
    Imagepath: str = ""
    # SEGAS 参数
    edit_concept_0: str = ""
    edit_concept_1: str = ""
    neg_guidance_0: bool = False
    neg_guidance_1: bool = False
    sega_val0: str = 'custom'
    sega_val1: str = 'custom'
    guidnace_scale_0: int = DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE
    guidnace_scale_1: int = DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE
    warmup_0: int = DEFAULT_WARMUP_STEPS
    warmup_1: int = DEFAULT_WARMUP_STEPS
    threshold_0: float = DEFAULT_THRESHOLD
    threshold_1: float = DEFAULT_THRESHOLD
    # 随机种子开关
    random_enabled: bool = False
          
client = Client("https://lpdoctor-ledits-jianjia.hf.space/",hf_token="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")

app = Flask(__name__)
CORS(app)
    
@app.route('/')
def index():
    return "demo test"

@app.route('/ledits-jianjia', methods=['GET','POST'])
def ledits():
    if request.method == 'POST':
        seed = DEFAULT_SEED
        # 如果ImageFile上传文件不符合规范
        if request.files.get('ImageFile') == None:
            return jsonify({'key':"error"})     

        # 保证文件是有效的!!!
        file_obj = request.files['ImageFile']
        if file_obj != None:
            file_obj.save('./tmp/'+ file_obj.filename)
            Entity.Imagepath = './tmp/'+ file_obj.filename
            client.predict(
		        api_name="/reset_do_inversion"
            )
            
            # Entity.Imagepath = client.predict(
            #     Entity.Imagepath,
            #     api_name="/crop_image"
            # )
            
            tar_prompt = client.predict(
                Entity.Imagepath,
                api_name='/obtain_target_prompt'
            )
            
             
            if request.form.get('Seed_switch') != None: 
                # 种子开关是False, 直接DDPM Inversion,不设置程序摇号
                if request.form.get('Seed_switch') == 'True' or request.form.get('Seed_switch') == 'true':
                    Entity.random_enabled = True
                    # 撒个种子
                    seed = client.predict(
                        DEFAULT_SEED,	# int | float  in 'Seed' Number component
                        Entity.random_enabled,	# bool  in 'Randomize seed' Checkbox component
                        api_name="/gen_seed"
                    )
                
            # DDPM Inversion process
            client.predict(
                Entity.Imagepath,	# str (filepath or URL to image) in 'Input Image' Image component
                seed,	# int | float  in 'Seed' Number component
                Entity.random_enabled,	# bool  in 'Randomize seed' Checkbox component
                "",	# str  in 'Source Prompt' Textbox component
                tar_prompt,	# str  in 'Describe your edited image (optional)' Textbox component
                DEFAULT_DIFFUSION_STEPS,	# int | float  in 'Num Diffusion Steps' Number component
                DEFAULT_SOURCE_GUIDANCE_SCALE,	# int | float  in 'Source Guidance Scale' Number component
                DEFAULT_SKIP_STEPS,	# int | float (numeric value between 0 and 60) in 'Skip Steps' Slider component
                DEFAULT_TARGET_GUIDANCE_SCALE,	# int | float (numeric value between 1 and 30) in 'Guidance Scale' Slider component
                api_name="/DDPM_load_and_invert"
            )
               
            # SEGAS Workplace
            if request.form.get('sega_val0') != "":  
                allocated_val0 = request.form.get('sega_val0')
                # 预处理
                # allocated_val0 = allocated_val0[0].strip('[').strip(']')  
                # list的第一个参数是type,第二个参数是prompt,第三个参数是removeButton?
                allocated_val0 = allocated_val0.split(',')
                # 默认长度为3 且 prompt不为空
                if len(allocated_val0) > 2 and allocated_val0[1] != '':
                    Entity.sega_val0 = allocated_val0[0]
                    Entity.edit_concept_0 = allocated_val0[1]
                    if allocated_val0[2] == 'True' or allocated_val0[2] == 'true':
                        Entity.neg_guidance_0 = True
                    allocated_para0 = client.predict(
                        Entity.sega_val0,
                        api_name="/LEDITS_SEGA_VAL1")
                    # 分配参数
                    Entity.guidnace_scale_0,Entity.warmup_0,Entity.threshold_0 = allocated_para0[0],allocated_para0[1],allocated_para0[2]
                
            if request.form.getlist('sega_val1') != "":  
                allocated_val1 = request.form.get('sega_val1')
                # 预处理
                # list的第一个参数是type,第二个参数是negativePrompt,第三个参数是remove Concept?
                allocated_val1 = allocated_val1.split(',')
                # negativePrompt不为空
                if len(allocated_val1) > 2 and allocated_val1[1] != '':
                    Entity.sega_val1 = allocated_val1[0]
                    Entity.edit_concept_1 = allocated_val1[1]
                    if allocated_val1[2] == 'True' or allocated_val1[2] == 'true':
                        Entity.neg_guidance_1 = True
                    allocated_para1 = client.predict(
                        Entity.sega_val1,
                        api_name="/LEDITS_SEGA_VAL2"
                        )
                    # 分配参数
                    Entity.guidnace_scale_1,Entity.warmup_1,Entity.threshold_1 = allocated_para1[0],allocated_para1[1],allocated_para1[2]
            
            
            resVal = client.predict(
                Entity.Imagepath,	# str (filepath or URL to image) in 'Input Image' Image component
                tar_prompt,	# str  in 'Describe your edited image (optional)' Textbox component
                DEFAULT_DIFFUSION_STEPS,	# int | float  in 'Num Diffusion Steps' Number component
                DEFAULT_SKIP_STEPS,	# int | float (numeric value between 0 and 60) in 'Skip Steps' Slider component
                DEFAULT_TARGET_GUIDANCE_SCALE,	# int | float (numeric value between 1 and 30) in 'Guidance Scale' Slider component 'tar_prompt的权重值'
                Entity.edit_concept_0,	# str  in 'Concept' Textbox component
                Entity.edit_concept_1,	# str  in 'Concept' Textbox component
                "",	# str  in 'Concept' Textbox component
                Entity.guidnace_scale_0,	# int | float (numeric value between 1 and 30) in 'Concept Guidance Scale' Slider component
                Entity.guidnace_scale_1,	# int | float (numeric value between 1 and 30) in 'Concept Guidance Scale' Slider component
                DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,	# int | float (numeric value between 1 and 30) in 'Concept Guidance Scale' Slider component
                Entity.warmup_0,	# int | float (numeric value between 0 and 50) in 'Warmup' Slider component
                Entity.warmup_1,	# int | float (numeric value between 0 and 50) in 'Warmup' Slider component
                DEFAULT_WARMUP_STEPS,	# int | float (numeric value between 0 and 50) in 'Warmup' Slider component
                Entity.neg_guidance_0,	# bool  in 'Remove Concept?' Checkbox component
                Entity.neg_guidance_1,	# bool  in 'Remove Concept?' Checkbox component
                DEFAULT_NEGATIVE_GUIDANCE,	# bool  in 'Remove Concept?' Checkbox component
                Entity.threshold_0,	# int | float (numeric value between 0.5 and 0.99) in 'Threshold' Slider component
                Entity.threshold_1,	# int | float (numeric value between 0.5 and 0.99) in 'Threshold' Slider component
                DEFAULT_THRESHOLD,	# int | float (numeric value between 0.5 and 0.99) in 'Threshold' Slider component
                seed,	# int | float  in 'Seed' Number component
                Entity.random_enabled,	# bool  in 'Randomize seed' Checkbox component
                "",	# str  in 'Source Prompt' Textbox component
                DEFAULT_SOURCE_GUIDANCE_SCALE,	# int | float  in 'Source Guidance Scale' Number component
                api_name="/LEDITS_edit"
            )
            
            
        # base64格式
        with open(resVal[0],"rb") as img_file:
          Response = base64.b64encode(img_file.read())
          return Response
        
        # with open(resVal[0],"rb") as img_file:
        #     return send_file(
        #         # 转二进制流
        #         io.BytesIO(img_file.read()),
        #         download_name = file_obj.filename,
        #         # mimetype = "image/png"
        #     )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

