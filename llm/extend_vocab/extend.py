import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
import sentencepiece as sp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', default=None, type=str)
parser.add_argument('--extend_vocab', default=None, type=str)
args = parser.parse_args()


origin_spm = sp_pb2.ModelProto()
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
origin_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

extend_spm = sp_pb2.ModelProto()
extend_vocab = sp.SentencePieceProcessor()
extend_vocab.Load(args.extend_vocab)
extend_spm.ParseFromString(extend_vocab.serialized_model_proto())

print(len(tokenizer),len(extend_vocab))
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)


token_set=set([p.piece for p in origin_spm.pieces])
print(f"Before:{len(token_set)}")

def p_piece(piece):
    new_p = sp_pb2.ModelProto().SentencePiece()
    new_p.piece = piece
    new_p.score = 0
    return new_p

news=list(map(p_piece,[p.piece for p in extend_spm.pieces if p.piece not in token_set]))
origin_spm.pieces.extend(news)

print(f"New: {len(origin_spm.pieces)}")

## Save
with open('extend.model', 'wb') as f:
    f.write(origin_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file='extend.model')

tokenizer.save_pretrained('./chinese_extend_tokenizer/')
os.remove('extend.model')
print('Done')