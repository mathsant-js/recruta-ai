# Curadoria documental do CKP02

Data da revisão: 06/10/2026.

## Escopo e método

A curadoria cobre cinco dimensões complementares do Recruta AI: processo de recrutamento
e seleção, privacidade, recrutamento justo, seleção por competências e diversidade. Foram
priorizados órgãos públicos e organismos internacionais. Cada PDF foi baixado do endereço
institucional, teve tipo, integridade, páginas, texto extraível e hash SHA-256 verificados.

O catálogo legível por máquina está em `data/manifest.json`. Os PDFs permanecem em
`data/raw/` sem conversão ou alteração, preservando o material original. A presença do
arquivo na base não transfere direitos autorais: o uso deve respeitar a nota de direitos de
cada item e manter a atribuição da fonte.

## Revisão de aplicabilidade

| ID | Contribuição para o RAG | Exemplos de perguntas sustentadas |
|---|---|---|
| DOC-01 | Estrutura o processo de recrutamento e seleção e seus responsáveis. | Quais etapas devem ser planejadas antes de iniciar uma seleção? Quais documentos e atores participam do processo? |
| DOC-02 | Aplica proteção de dados à fase pré-contratual e ao uso de IA. | Quais dados de candidatos podem ser excessivos? Quando decisões automatizadas exigem revisão humana? |
| DOC-03 | Define princípios de recrutamento justo e responsabilidades institucionais. | Quais deveres cabem às empresas para prevenir recrutamento abusivo? Quem deve arcar com custos de recrutamento? |
| DOC-04 | Traduz boas práticas para perfil de vaga, entrevistas, diversidade e competências. | Como planejar uma entrevista por competências? Como reduzir subjetividade na definição do perfil? |
| DOC-05 | Traz evidências sobre diversidade, inclusão e barreiras em processos seletivos. | Quais barreiras afetam grupos vulnerabilizados? Que práticas favorecem inclusão e permanência? |

## Cobertura cruzada

As fontes permitem perguntas que exigem mais de um documento. Por exemplo:

- combinar DOC-02 e DOC-05 para discutir coleta de dados e riscos de exclusão em triagens;
- combinar DOC-01 e DOC-04 para comparar planejamento geral com práticas de entrevista;
- combinar DOC-03, DOC-04 e DOC-05 para avaliar transparência, equidade e inclusão;
- combinar DOC-02 e DOC-03 para tratar revisão humana e direitos durante o recrutamento.

## Limitações identificadas

- DOC-01 e DOC-04 foram escritos para o setor público; recomendações devem ser adaptadas
  antes de serem generalizadas para empresas privadas.
- DOC-03 usa português europeu e dedica parte do conteúdo a trabalhadores migrantes; a
  recuperação deve preservar esse contexto.
- DOC-05 é uma pesquisa descritiva, não uma norma jurídica.
- DOC-02 é um relatório consultivo recente. Ele não substitui leitura da LGPD, orientação
  jurídica ou normas posteriores.
- A base não contém currículos nem dados pessoais reais e não deve ser usada para decidir
  automaticamente sobre candidatos.

## Critério de aceite da fase 1

- [x] Cinco documentos reais do domínio.
- [x] Fontes oficiais ou institucionais e URLs registradas.
- [x] Arquivos íntegros, legíveis e com texto extraível.
- [x] Hash, metadata e justificativa de inclusão registrados.
- [x] Capacidade de sustentar perguntas não triviais e cruzadas revisada.
- [x] Nenhum currículo ou dado pessoal real incluído.

A fase 1 está pronta para a fase 2. A próxima etapa deve validar o manifesto, implementar
loaders e comparar os primeiros parâmetros de chunking sem alterar manualmente os PDFs.
