def extract(text, word_count, sentence_count):
    if word_count<3 or sentence_count<1: return {'F06':float('nan'),'F07':float('nan'),'F08':float('nan')}
    try:
        import textstat
    except Exception:
        return {'F06':float('nan'),'F07':float('nan'),'F08':float('nan')}
    return {'F06':float(textstat.flesch_reading_ease(text)),'F07':float(textstat.flesch_kincaid_grade(text)),'F08':float(textstat.gunning_fog(text))}
