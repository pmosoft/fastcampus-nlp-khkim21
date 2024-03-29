import sys



if __name__ == "__main__":
    #ref_fn = sys.argv[1]
    #f = open(ref_fn, 'r')
    fr1 = open("review.sorted.uniq.refined.tsv.text","r", encoding='utf-8')
    fr2 = open("review.sorted.uniq.refined.tsv.text.tok","r", encoding='utf-8')
    fw = open("review.sorted.uniq.refined.tsv.text.tok2","w", encoding='utf-8')
    for ref in fr1:
        ref_tokens = ref.strip().split(' ')
        input_line = fr2.readline().strip()

        if input_line != "":
            tokens = input_line.split(' ')

            idx = 0
            buf = []

            # We assume that stdin has more tokens than reference input.
            for ref_token in ref_tokens:
                tmp_buf = []

                while idx < len(tokens):
                    if tokens[idx].strip() == '':
                        idx += 1
                        continue

                    tmp_buf += [tokens[idx]]
                    idx += 1

                    if ''.join(tmp_buf) == ref_token:
                        break

                if len(tmp_buf) > 0:
                    buf += [STR + tmp_buf[0].strip()] + tmp_buf[1:]

            #sys.stdout.write(' '.join(buf) + '\n')
            fw.write(' '.join(buf) + '\n')
        else:
            #sys.stdout.write('\n')
            fw.write('\n')

    fr1.close()