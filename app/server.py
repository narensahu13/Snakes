from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/open?id=1r6l9lfrkQno3fmMlT2d_WMQ629bcbQWI'
export_file_name = 'stage-2.pth'

classes = ['Anaconda',"Baird's Rat Snake",'Ball Python','Beaked Sea Snake',"Belcher's Sea Snake",'Black Mamba','Black Rat Snake',
          'Burmese Python','Bushmaster','California Kingsnake','Cape Cobra','Carpet Python','Copperheads','Corn Snake',
           'Desert Kingsnake','Egyptian Cobra','Emerald Tree Boa','Gaboon Viper','Green Mamba','Green Tree Python',
           'Grey-banded Kingsnake','Indian Cobra','Inland Taipan','King Cobra','King Ratsnake','Mozambique Spitting Cobra',
           'Olive Sea Snake','Rattlesnakes','Red Milk Snake','Red Spitting Cobra','Reticulated Python','Rhinoceros Viper',
          'Ribbon Snake','Rosy Boa','Rough Green Snake',"Russell's Viper",'Saw-scaled Viper','Scarlet Snake','Spiny-headed Sea Snake',
           'Spiny-tailed Sea Snake','Temple Viper','The African Rock Python','The Arabian Gulf Sea Snake','Turtlehead Sea Snake',
           'Water Moccasin','Western Coachwhip','White-lipped Python','Yellow Snake','Yellow-bellied Sea Snake','Yellow-lipped Sea Krait']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
