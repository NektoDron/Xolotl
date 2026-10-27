#include "m31_prompt_cache.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace strata::m31;
using namespace strata::m31::prompt_cache;
static void check(bool ok,const char * reason){if(!ok)throw std::runtime_error(reason);}
int main(int argc,char ** argv){
 try {
    if(argc!=2)throw std::runtime_error("usage: test tiny.gguf");
    for (bool portable : {false,true}) {
        Sha256 empty(portable);check(hex(empty.finish())=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","empty SHA256 KAT");
        Sha256 abc(portable);abc.add("a",1);abc.add("bc",2);check(hex(abc.finish())=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","abc SHA256 KAT");
    }
    for(size_t size : {1,55,56,63,64,65,127,128,129,10000}){
        std::vector<uint8_t> bytes(size);for(size_t i=0;i<size;++i)bytes[i]=uint8_t(i*17+3);
        Sha256 accelerated,portable(true);
        accelerated.add(bytes.data(),size);
        for(size_t i=0;i<size;){size_t count=std::min(size-i,size_t(13));portable.add(bytes.data()+i,count);i+=count;}
        check(accelerated.finish()==portable.finish(),"chunked portable/accelerated SHA mismatch");
    }
    std::vector<int32_t> ids{5,7,11,17}, map{-1,-1,-1,-1}, positions{0,1,2,3};
    std::vector<float> emb{0.f,1.f};
    const auto key=input_key("engine-one",ids,map,emb,positions);
    check(key!=input_key("engine-two",ids,map,emb,positions),"model/config omitted from key");
    emb[0]=-0.f;check(key!=input_key("engine-one",ids,map,emb,positions),"embedding bytes omitted from key");emb[0]=0.f;
    map[0]=0;check(key!=input_key("engine-one",ids,map,emb,positions),"visual map omitted from key");map[0]=-1;
    positions[0]=1;check(key!=input_key("engine-one",ids,map,emb,positions),"positions omitted from key");positions[0]=0;
    ids[0]=6;check(key!=input_key("engine-one",ids,map,emb,positions),"text IDs omitted from key");ids[0]=5;
    check(entry_key(key,{{0,4}})!=entry_key(key,{{0,2},{2,2}}),"actual chunk recipe omitted from entry key");
    bool rejected=false;try{Cache too_large(1024ull*1024*1024+1);}catch(...){rejected=true;}check(rejected,"unbounded host cache accepted");
    Engine engine;Config cfg;cfg.context=32;cfg.max_chunk=32;cfg.threads=2;
    std::string error;
    auto engine_check=[&](bool ok){if(!ok)throw std::runtime_error(error);};
    engine_check(engine.init(argv[1],cfg,error));
    engine_check(engine.forward(ids.data(),int(ids.size()),error));
    std::vector<float> prefix_logits;
    engine_check(engine.logits(3,1,prefix_logits,error));
    Engine::PrefixCheckpointPtr checkpoint;
    engine_check(engine.checkpoint_prefix(engine.prefix_checkpoint_bytes(),checkpoint,error));
    check(checkpoint->position()==4&&checkpoint->bytes()==engine.geometry().kv_bytes(4),"checkpoint geometry mismatch");
    const int32_t tail[]={3,9,2};
    std::vector<std::vector<float>> reference;
    for(int32_t token:tail){std::vector<float> logits;engine_check(engine.forward(&token,1,error)&&engine.logits(0,1,logits,error));reference.push_back(std::move(logits));}
    const auto stable_pos=engine.position();
    Engine::PrefixCheckpointPtr failed=checkpoint;
    check(!engine.checkpoint_prefix(1,failed,error)&&engine.position()==stable_pos&&failed==checkpoint,"budget rejection modified state/output");
    check(!engine.restore_prefix({},error)&&engine.position()==stable_pos,"null checkpoint modified state");
    Engine other;engine_check(other.init(argv[1],cfg,error));
    check(!other.restore_prefix(checkpoint,error)&&other.position()==0,"foreign checkpoint accepted");
    engine.reset();int32_t changed[]={2,8,12,23,6,29,15};engine_check(engine.forward(changed,7,error));
    engine_check(engine.restore_prefix(checkpoint,error)&&engine.position()==4);
    std::vector<float> no_chunk;check(!engine.logits(0,1,no_chunk,error),"restored prefix exposed stale last-chunk logits");
    for(size_t i=0;i<3;++i){std::vector<float> logits;engine_check(engine.forward(tail+i,1,error)&&engine.logits(0,1,logits,error));check(logits.size()==reference[i].size()&&!std::memcmp(logits.data(),reference[i].data(),logits.size()*sizeof(float)),"restored continuation is not bit-identical");}
    Entry a;a.input=key;a.checkpoint=checkpoint;a.logits=prefix_logits;a.recipe={{0,4}};
    Entry b=a;b.input[0]^=1;Entry c=a;c.input[0]^=2;
    Cache cache(2*a.bytes());check(cache.put(a)&&cache.put(b),"cache insertion failed");
    check(cache.find(a.input),"cache miss on exact key");check(cache.put(c),"LRU insertion failed");
    check(!cache.find(b.input)&&cache.find(a.input)&&cache.find(c.input),"LRU did not evict least recent entry");
    check(cache.bytes()<=cache.budget()&&cache.entries()==2,"host cache accounting exceeded budget");
    Entry invalid=a;invalid.recipe={{1,4}};check(!cache.put(invalid),"invalid chunk recipe accepted");
    invalid=a;invalid.recipe={{0,2},{2,2}};check(!cache.put(invalid),"multi-chunk cache entry accepted");
    Cache tiny(a.bytes()-1);check(!tiny.put(a)&&tiny.entries()==0,"oversize entry stored");
    Cache disabled;check(!disabled.put(a)&&disabled.entries()==0,"disabled cache stored an entry");
    std::cout<<"PASS: SHA256 KAT/chunking/fast-fallback; exact content keys; owner-bound KV checkpoint/restore/continuation bitwise; rejection safety; bounded LRU/recipe\n";
    return 0;
 }catch(const std::exception & e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
