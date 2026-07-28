from .adult import load_and_preprocess_adult
from .compas import load_and_preprocess_compas
from .dropout import load_and_preprocess_dropout
from .intersectional_bias import load_and_preprocess_intersectional_bias
from .loaders_datasus import load_and_preprocess_sih, load_and_preprocess_sim, load_and_preprocess_sinasc
from .loaders_cadunico import load_and_preprocess_cadunico

# Dictionary mapping dataset names to their loader functions and relevant metadata
DATASETS = {
    'Adult 🇺🇸': {
        'loader': load_and_preprocess_adult,
        'target': 'income',
        'favorable_val': 1,
        'protected_attributes': ['sex', 'race', 'age_group', 'education_group', 'relationship'],
        'description': "Domínio: Censo Demográfico / Renda (EUA). Avalia se a renda anual excede $50K. Atributos sensíveis incluem sexo, raça e escolaridade. Grupos frequentemente privilegiados: Homens Brancos."
    },
    'COMPAS 🇺🇸': {
        'loader': load_and_preprocess_compas,
        'target': 'two_year_recid',
        'favorable_val': 0, # Favorable is NOT recidivating
        'protected_attributes': ['sex', 'race', 'age_group'],
        'description': "Domínio: Justiça Criminal (EUA). Estima o risco de reincidência criminal em 2 anos. Atributos sensíveis incluem raça e sexo. O viés estrutural frequentemente superestima o risco para réus afro-americanos."
    },
    'Dropout 🇵🇹': {
        'loader': load_and_preprocess_dropout,
        'target': 'Target',
        'favorable_val': 1, # Graduate
        'protected_attributes': ['Gender', 'Age_Group', 'Mother_Qualification_Group'],
        'description': "Domínio: Educação (Portugal). Prevê a evasão ou formatura de estudantes universitários. Atributos sensíveis abrangem gênero, idade e qualificação dos pais. Curiosidade: mulheres apresentam taxas naturais de retenção significativamente maiores."
    },
    'Intersectional Bias 🌐': {
        'loader': load_and_preprocess_intersectional_bias,
        'target': 'diagnosis',
        'favorable_val': 1, 
        'protected_attributes': ['race', 'sex'],
        'description': "Domínio: Saúde (Benchmark Sintético OpenML). Desenvolvido especificamente para auditar diagnósticos clínicos com disparidades interseccionais (ex: viés oculto exacerbado na interseção entre raça e sexo)."
    },
    'SIH (DATASUS) 🇧🇷': {
        'loader': load_and_preprocess_sih,
        'target': 'desfecho',
        'favorable_val': 'Alta', 
        'protected_attributes': ['sexo', 'raca_cor'],
        'description': "Domínio: Saúde Pública (Brasil). Sistema de Informações Hospitalares. Avalia o desfecho da internação (Alta vs. Óbito). Revela disparidades na qualidade da assistência segundo raça/cor e sexo."
    },
    'SIM (DATASUS) 🇧🇷': {
        'loader': load_and_preprocess_sim,
        'target': 'tipo_obito',
        'favorable_val': 'Não Evitável', 
        'protected_attributes': ['sexo', 'raca_cor'],
        'description': "Domínio: Saúde Pública (Brasil). Sistema de Informações sobre Mortalidade. Analisa se o óbito era evitável por intervenções do SUS. Exibe iniquidades de acesso e mortalidade precoce focada em raça/cor."
    },
    'SINASC (DATASUS) 🇧🇷': {
        'loader': load_and_preprocess_sinasc,
        'target': 'desfecho_nascimento',
        'favorable_val': 'Normal',
        'protected_attributes': ['raca_cor_mae', 'idade_mae', 'escolaridade_mae'],
        'description': "Domínio: Saúde Materno-Infantil (Brasil). Sistema de Informações sobre Nascidos Vivos. Foca em desfechos como prematuridade e baixo peso ao nascer, controlando determinantes sociais da mãe."
    },
    'CadÚnico 🇧🇷': {
        'loader': load_and_preprocess_cadunico,
        'target': 'pobreza_extrema',
        'favorable_val': 0, # Not in extreme poverty
        'protected_attributes': ['raca_cor', 'sexo', 'escolaridade'],
        'description': "Domínio: Assistência Social (Brasil). Amostra do Cadastro Único (2012-2016). Avalia se a renda per capita familiar está abaixo da linha de extrema pobreza (R$ 77,00 na época). Fortemente marcado pelo racismo e sexismo estrutural."
    }
}
