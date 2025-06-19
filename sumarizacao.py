# Tokenizer 
from transformers import T5Tokenizer
import re
# PyTorch model 
from transformers import T5Model, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
model_name = 'recogna-nlp/ptt5-base-summ'
tokenizer = T5Tokenizer.from_pretrained(token_name )
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = '''
Trabalhadores de serviços públicos que escavavam uma vala para expandir a rede de gás natural em Lima, no Peru, encontraram uma múmia pré-inca de aproximadamente mil anos. O corpo estava enterrado em uma vala rasa, a cerca de meio metro da superfície. (Veja a imagem acima)

Achado arqueológico no meio da cidade
A múmia foi descoberta na capital peruana e permaneceu por anos despercebida, mesmo com o crescimento urbano da região. Os trabalhadores faziam a instalação de um gasoduto na semana passada quando encontraram os restos mortais.

De acordo com os arqueólogos chamados para acompanhar a escavação, o corpo estava enterrado sentado, envolto em um fardo e ainda preservava parte dos cabelos.

“Encontramos restos mortais e evidências de que pode ter havido um enterro pré-hispânico”, afirmou o arqueólogo Arturo Aliaga, que acompanha o monitoramento arqueológico na região.

Quem era essa múmia?
Segundo Jesús Bahamonde, diretor do plano de monitoramento arqueológico da região metropolitana de Lima, a múmia encontrada na semana passada seria de um corpo pertencido a uma sociedade de pescadores da cultura Chancay, que habitou a costa central do Peru entre os anos 1.000 e 1.470 d.C.

Lima: uma cidade moderna sobre sítios arqueológicos
Lima, capital do Peru, está localizada em um vale irrigado por três rios vindos dos Andes. A região foi ocupada por civilizações humanas milhares de anos antes da chegada dos colonizadores espanhóis em 1535.

Hoje, com cerca de 10 milhões de habitantes, Lima abriga mais de 400 sítios arqueológicos, muitos deles integrados ao ambiente urbano atual.

“É muito comum encontrar vestígios arqueológicos na costa peruana, incluindo Lima, principalmente elementos funerários: tumbas, sepultamentos e, entre estes, indivíduos mumificados”, explicou Pieter Van Dalen, decano do Colégio de Arqueólogos do Peru.

Como essas múmias são preservadas?
Van Dalen, decano do Colégio de Arqueólogos do Peru, que não participou diretamente da descoberta, explicou que muitas múmias na costa do Peru são mumificadas naturalmente, graças ao clima árido e quente, que desidrata a pele.

Outros corpos, segundo ele, passaram por processos de mumificação cultural, prática comum entre povos pré-incas. Nesse tipo de sepultamento, os indivíduos geralmente são encontrados sentados, com as mãos cobrindo o rosto, envolvidos em tecidos que compõem o fardo funerário.
'''

def remover_tags(texto):
    texto_limpo = re.sub(r'</?s>|<pad>', '', texto)
    return texto_limpo.strip()

def contar_palavras(texto):
    palavras = texto.split()
    return len(palavras)

print("Quantidade de palavras do texto original: " + str(contar_palavras(text)))
inputs = tokenizer.encode(text, max_length=1024, truncation=True, return_tensors='pt')
summary_ids = model.generate(inputs, max_length=512, min_length=100, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
summary = tokenizer.decode(summary_ids[0])
summary = remover_tags(summary)
print("Quantidade de palavrs do resumo: " + str(contar_palavras(summary)))
print("Resumo: \n" + summary)
