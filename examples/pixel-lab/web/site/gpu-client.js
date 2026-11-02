export class GpuClient {
  constructor(progress = () => {}) {
    this.worker = new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
    this.nextId = 1;
    this.pending = new Map();
    this.worker.onmessage = ({data}) => {
      if(data.fatal) {this.close(new Error(data.fatal));progress(data.fatal);return;}
      if(data.status!==undefined||data.download!==undefined||data.completed||data.mask!==undefined) {progress(data);return;}
      const pending=this.pending.get(data.id);
      if(!pending) return;
      this.pending.delete(data.id);clearTimeout(pending.timer);
      if(data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    this.worker.onerror = event => this.close(new Error(event.message||'The GPU worker failed.'));
    this.worker.onmessageerror = () => this.close(new Error('Unable to communicate with the GPU worker.'));
  }

  request(type,payload={},transfer=[],timeout=60000) {
    if(!this.worker) return Promise.reject(new Error('The GPU worker has stopped.'));
    const id=this.nextId++;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>this.close(new Error(`${type} timed out. Stop and restart the camera to retry.`)),timeout);
      this.pending.set(id,{resolve,reject,timer});
      try {this.worker.postMessage({id,type,payload},transfer);}
      catch(error) {clearTimeout(timer);this.pending.delete(id);reject(error);}
    });
  }

  close(error=new Error('GPU processing stopped.')) {
    this.worker?.terminate();this.worker=null;
    for(const pending of this.pending.values()) {clearTimeout(pending.timer);pending.reject(error);}
    this.pending.clear();
  }
}
