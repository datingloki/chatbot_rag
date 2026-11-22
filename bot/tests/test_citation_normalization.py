import pytest
from tests.conftest import CHUNK,ask,make_bot,reply
from polybot.clients.llm import ScriptedLLM
from polybot.schemas import ReplyKind

@pytest.mark.parametrize('citation',[f'[{CHUNK.chunk_id}]',f'  {CHUNK.chunk_id}  ',f'[ {CHUNK.chunk_id} ]'])
def test_exact_evidence_citation_with_display_wrapper_is_accepted(citation):
 bot=make_bot(llm=ScriptedLLM(reply('answer','Замените материал на закладке «Материалы».',[citation])))
 result=ask(bot,'Как учесть замену материала?')
 assert result.kind is ReplyKind.ANSWER
 assert result.sources[0].url==CHUNK.url
 assert result.llm.calls==1

@pytest.mark.parametrize('citation',['[faq:absent#0]','[[faq:poly2-261#0]]','faq:poly2-261',{'id':'faq:poly2-261#0'}])
def test_normalization_never_admits_unknown_or_non_string_source(citation):
 bot=make_bot(llm=ScriptedLLM(reply('answer','Ответ.',[citation])))
 result=ask(bot,'Вопрос')
 assert result.kind is ReplyKind.FALLBACK
 assert result.sources==[]
 assert result.reason in ('validation_unknown_citation','validation_bad_schema')
